# model based on https://arxiv.org/pdf/1905.09263.pdf
# supplementary logic is partly based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html

import torch
from torch import nn, Tensor
import math
from torch.nn.utils.rnn import pad_sequence
from hw_tts.aligner import Batch


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        # d_model - embedding size
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] expected
            but ours is of size([batch_size, seq_len, embedding_dim])
        """
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x.transpose(0, 1))


"""
class Attention(nn.Module):
    # based on https://jalammar.github.io/illustrated-transformer/
    def __init__(self, d_model: int, d_hid: int):
        super().__init__()
        self.WQ = nn.Linear(d_model, d_hid, bias=False)
        self.WK = nn.Linear(d_model, d_hid, bias=False)
        self.WV = nn.Linear(d_model, d_hid, bias=False)
        self.d_hid = d_hid

    def forward(self, x, mask=None):
        # x: [batch, seq_len, emb_sz]
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)
        Z_pred = torch.matmul(Q, K.transpose(1, 2)) / self.d_hid ** 0.5
        if mask is not None:
            Z_pred = Z_pred.masked_fill(mask == 0, -9e15)
        Z_prob = nn.functional.softmax(Z_pred, dim=-1)
        Z = torch.matmul(Z_prob, V)
        if self.training:
            return Z, None
        return Z, Z_prob
"""


class MultiHeadAttention(nn.Module):
    # based on https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    def __init__(self, d_model: int, nhead: int, d_hid: int):
        super().__init__()
        assert d_model % nhead == 0
        self.QKV = nn.Linear(d_model, 3 * d_hid, bias=False)
        self.combiner = nn.Linear(d_hid, d_model, bias=False)
        self.nhead = nhead
        self.h_dim = d_model // nhead
        self.d_hid = d_hid
        self.initialise_weights()

    def initialise_weights(self):
        gain = (2/5)**0.5
        nn.init.xavier_normal_(self.QKV.weight, gain)
        nn.init.xavier_normal_(self.combiner.weight, gain)

    @staticmethod
    def calc_attention(Q, K, V, mask=None):
        d_k = Q.size(-1)
        Z_pred = torch.matmul(Q, K.transpose(-1, -2)) / d_k ** 0.5
        if mask is not None:
            Z_pred = Z_pred.masked_fill((mask == 0)[:, None, None, :], -9e15)
        Z_prob = nn.functional.softmax(Z_pred, dim=-1)
        Z = torch.matmul(Z_prob, V)
        return Z, Z_prob

    def forward(self, x, mask=None):
        # x:[batch, seq_len, emb_sz]
        b_sz, seq_ln, emb_sz = x.size()
        QKV = self.QKV(x)

        # now we want [b_sz, seq_len, 3*emb_sz] -> [b_sz, n_head, seq_ln, 3*(emb_sz//n_head)]
        QKV = QKV.reshape(b_sz, seq_ln, self.nhead, 3 * self.h_dim).permute(0, 2, 1, 3)
        Q, K, V = QKV.chunk(3, dim=-1)

        vals, attnt = self.calc_attention(Q, K, V, mask)
        vals = vals.permute(0, 2, 1, 3).reshape(b_sz, seq_ln, self.d_hid)
        out = self.combiner(vals)

        if self.training:
            return nn.ReLU()(out), None
        # attnt is of size [batch_sz, n_head, seq_len, seq_len]
        # for batch_sz=1 output [ n_head, seq_len, seq_len] is expected
        return nn.ReLU()(out), attnt.transpose(0, 1).squeeze()


class ConvNet1d(nn.Module):
    def __init__(self, d_model: int, d_hid_ker: int, kernel_sz: int):
        # two layered convnet from paper
        super().__init__()
        convs = []
        convs.append(nn.Conv1d(d_model, d_hid_ker, kernel_sz, padding='same'))
        convs.append(nn.ReLU())
        convs.append(nn.Conv1d(d_hid_ker, d_model, kernel_sz, padding='same'))
        convs.append(nn.ReLU())
        self.net = nn.ModuleList(convs)
        self.init_weights()

    def init_weights(self):
        gain = (2 / 5) ** 0.5
        nn.init.xavier_normal_(self.net[0].weight, gain)
        if self.net[0].bias is not None:
            nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_normal_(self.net[2].weight, gain)
        if self.net[2].bias is not None:
            nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        # x: [batch, sq_len, emb_sz]
        out = x.transpose(1, 2)
        for elem in self.net:
            out = elem(out)
        return out.transpose(1, 2)


class FSFFTBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 kernel_sz: int, filter_sz: int, dropout: float = 0.5,
                 pre_layer_norm: bool = True):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.mhattention = MultiHeadAttention(d_model, nhead, d_hid)
        self.convnet = ConvNet1d(d_model, filter_sz, kernel_sz)
        self.mhead_norm = torch.nn.LayerNorm(d_model)
        self.conv_norm = torch.nn.LayerNorm(d_model)
        self.pln = pre_layer_norm

        # TODO: ?????? ???????? ?????????? ????????, ?????????????? ?????? ??????????. ???????? ?????????????????? ?????????? ?? ?? ???????????? ??????????
        # ???????? ???????????? ?????? ?? ???????????????????? ???????????????????????? ????????????????????

    def forward(self, x, mask=None):
        # x:[batch, seq_len, emb_sz]
        out, _ = self.mhattention(x, mask)
        if self.pln:
            out = x + self.mhead_norm(out)
        else:
            out = self.mhead_norm(x + out)
        out = self.dropout(out)

        out_sec = self.convnet(out)
        if self.pln:
            out = out + self.conv_norm(out_sec)
        else:
            out = self.conv_norm(out + out_sec)
        out = self.dropout(out)
        return out


class DurationPredictor(nn.Module):
    def __init__(self, d_model: int, filter_sz: int, kernel_sz: int, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            ConvNet1d(d_model, filter_sz, kernel_sz),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            ConvNet1d(d_model, filter_sz, kernel_sz),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x[batch_sz, seq_len, emb_sz]
        out = self.net(x)
        return out


class LengthRegulator(nn.Module):
    def __init__(self, d_model: int, filter_sz: int, kernel_sz: int, dropout: float = 0.5, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.dpred = DurationPredictor(d_model, filter_sz, kernel_sz, dropout)

    def forward(self, y, x: Batch, device):
        # y: encoder output, [batch_sz, seq_ln, emb_sz]
        # x: initial batch with wav info

        preds = self.dpred(y)
        # preds: [batch_sz, seq_ln, 1]
        if self.training:
            enlarged = []
            new_lns = []
            for i in range(y.size(0)):
                rel_lengths = torch.from_numpy(x.alignment[i]).to(device)
                cur_enlargement = torch.repeat_interleave(y[i, :x.token_lengths[i], :], rel_lengths, dim=0)
                # firstly i want to restore true number of frames for melspec, thus i add zeros
                enlarged.append(cur_enlargement)
            enlarged = pad_sequence(enlarged, batch_first=True, padding_value=0.)
            return enlarged, preds.squeeze(-1)
        else:
            lns = torch.round(self.alpha * torch.exp(preds)).int().squeeze(-1)
            enlarged = []
            for i in range(y.size(0)):
                enlarged.append(
                    torch.repeat_interleave(y[i, :x.token_lengths[i], :], lns[i, :x.token_lengths[i]], dim=0))
            enlarged = pad_sequence(enlarged, batch_first=True, padding_value=0.)

            return enlarged, None


class FastSpeech(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 kernel_sz: int, filter_sz: int, dur_pred_filter_sz: int,
                 nlayers: int, n_mels: int, dropout: float = 0.5, phon_max: int = 300, frame_max: int = 5000,
                 alpha: float = 1.0):
        # phon_max: maximum length of phoneme input
        # frame_max: maximum length of frames after length regulator

        # formally we should support different parameters for encoder/decoder parts,
        # but we omit this and use similar parameters in both parts
        super().__init__()
        self.d_model = d_model

        self.phoneme_embedding = nn.Embedding(ntoken, d_model)

        self.phon_pos_enc = PositionalEncoding(d_model, dropout, phon_max)

        first_half = []
        for i in range(nlayers):
            first_half.append(FSFFTBlock(d_model, nhead, d_hid, kernel_sz, filter_sz, dropout))
        self.encoder = nn.ModuleList(first_half)

        self.length_regulator = LengthRegulator(d_model, dur_pred_filter_sz, kernel_sz, dropout, alpha)

        self.mult_pos_enc = PositionalEncoding(d_model, dropout, frame_max)

        sec_half = []
        for i in range(nlayers):
            sec_half.append(FSFFTBlock(d_model, nhead, d_hid, kernel_sz, filter_sz, dropout))
        self.decoder = nn.ModuleList(sec_half)

        self.predictor = nn.Linear(d_model, n_mels)

    def forward(self, x: Batch, device: torch.device):
        # x: Batch class.
        # x.tokens: [batch_sz, seq_ln]

        if self.training:
            mask = torch.zeros(x.tokens.size()).to(device)
            for i in range(x.tokens.size(0)):
                mask[i, :x.token_lengths[i]] = 1
        else:
            mask = torch.ones(x.tokens.size()).to(device)

        out = self.phoneme_embedding(x.tokens) * self.d_model ** 0.5
        out = self.phon_pos_enc(out)
        for elem in self.encoder:
            out = elem(out, mask)

        out, pred_log_len = self.length_regulator(out, x, device)
        out = self.mult_pos_enc(out.to(device))

        if self.training:
            mask = torch.zeros(out.size()[:2]).to(device)
            for i in range(out.size(0)):
                mask[i, :x.alignment[i].sum()] = 1
        else:
            mask = torch.ones(out.size()[:2]).to(device)
        for elem in self.decoder:
            out = elem(out, mask)

        out = self.predictor(out)

        if self.training:
            return out, pred_log_len

        return out



