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
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Attention(nn.Module):
    # based on https://jalammar.github.io/illustrated-transformer/
    def __init__(self, d_model: int, d_hid: int):
        super().__init__()
        self.WQ = nn.Linear(d_model, d_hid)
        self.WK = nn.Linear(d_model, d_hid)
        self.WV = nn.Linear(d_model, d_hid)
        self.d_hid = d_hid

    def forward(self, x):
        # x: [batch, seq_len, emb_sz]
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)
        Z = nn.functional.softmax(torch.matmul(Q, K.transpose(1, 2))/self.d_hid**0.5, dim=-1)
        Z = torch.matmul(Z, V)
        return Z


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_hid: int):
        super().__init__()
        self.heads = nn.ModuleList()
        for i in range(nhead):
            self.heads.append(Attention(d_model, d_hid))
        self.combiner = nn.Linear(nhead*d_hid, d_model)
        self.nhead = nhead

    def forward(self, x):
        # x:[batch, seq_len, emb_sz]
        Zs = []
        for i in range(self.nhead):
            Zs.append(self.heads[i](x))
        Zs = torch.cat(Zs, dim=-1)
        out = self.combiner(Zs)
        return nn.ReLU()(out)


class ConvNet1d(nn.Module):
    def __init__(self, d_model: int, d_hid_ker:int, kernel_sz: int):
        # two layered convnet from paper
        super().__init__()
        convs = []
        convs.append(nn.Conv1d(d_model, d_hid_ker, kernel_sz, padding='same'))
        convs.append(nn.ReLU())
        convs.append(nn.Conv1d(d_hid_ker, d_model, kernel_sz, padding='same'))
        convs.append(nn.ReLU())
        self.net = nn.Sequential(*convs)

    def forward(self, x):
        # x: [batch, sq_len, emb_sz]
        out = self.net(x.transpose(1, 2))
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

        # TODO: все такой какой норм, инстанс или лейер. надо поправить здесь и в другом месте
        # Пока сделал как в реализации трансформера торчовской

    def forward(self, x):
        # x:[batch, seq_len, emb_sz]
        out = self.mhattention(x)
        if self.pln:
            out = x + self.mhead_norm(out)
        else:
            out = self.mhead_norm(x+out)
        out = self.dropout(out)

        out_sec = self.convnet(out)
        if self.pln:
            out = out + self.mhead_norm(out_sec)
        else:
            out = self.mhead_norm(out+out_sec)
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

    def forward(self, y, x: Batch, device, melspec, galigner):
        # y: encoder output, [batch_sz, seq_ln, emb_sz]
        # x: initial batch with wav info

        preds = self.dpred(y)
        # preds: [batch_sz, seq_ln, 1]
        if self.training:
            durs = galigner(x.waveform, x.waveform_length, x.transcript)
            enlarged = []
            ground_truth_lns = []
            new_lns = []
            for i in range(y.size(0)):
                rel_lengths = durs[i]/durs[i].sum()
                true_ln = melspec(x.waveform[i, :x.waveform_length[i]]).size(0)
                approx_lns = torch.round(true_ln*rel_lengths).int().to(device)
                cur_enlargement = torch.repeat_interleave(y[i, :x.token_lengths[i], :], approx_lns[1:-1], dim=0)
                # firstly i want to restore true number of frames for melspec, thus i add zeros
                true_sz = torch.full((approx_lns.sum(), y.size(2)), Batch.pad_value)
                true_sz[approx_lns[0]:approx_lns[-1]] = cur_enlargement
                ground_truth_lns.append(approx_lns[1:-1])
                enlarged.append(true_sz)
                new_lns.append(approx_lns.sum().item())
            enlarged = pad_sequence(enlarged, batch_first=True, padding_value=0.)
            return enlarged, new_lns, preds.squeeze(-1), ground_truth_lns

        else:
            lns = torch.round(self.alpha * torch.exp(preds)).int().squeeze(-1)
            enlarged = []
            for i in range(y.size(0)):
                enlarged.append(
                    torch.repeat_interleave(y[i, :x.token_lengths[i], :], lns[i, :x.token_lengths[i]], dim=0))
            enlarged = pad_sequence(enlarged, batch_first=True, padding_value=Batch.pad_value)

            return enlarged, None, None, None


class FastSpeech(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 kernel_sz: int, filter_sz: int, dur_pred_filter_sz: int,
                 nlayers: int, n_mels: int, dropout: float = 0.5, phon_max: int = 5000, frame_max: int = 5000,
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
        self.encoder = nn.Sequential(*first_half)

        self.length_regulator = LengthRegulator(d_model, dur_pred_filter_sz, kernel_sz, dropout, alpha)

        self.mult_pos_enc = PositionalEncoding(d_model, dropout, frame_max)

        sec_half = []
        for i in range(nlayers):
            sec_half.append(FSFFTBlock(d_model, nhead, d_hid, kernel_sz, filter_sz, dropout))
        self.decoder = nn.Sequential(*sec_half)

        self.predictor = nn.Linear(d_model, n_mels)

    def forward(self, x: Batch, device: torch.device, melspec: nn.Module, galigner: nn.Module):
        # x: Batch class.
        # x.tokens: [batch_sz, seq_ln]
        out = self.phoneme_embedding(x.tokens) * self.d_model**0.5
        out = self.phon_pos_enc(out)
        out = self.encoder(out)

        out, new_lns, pred_log_len, true_log_len = self.length_regulator(out, x, device, melspec, galigner)

        out = self.mult_pos_enc(out.to(device))
        out = self.decoder(out)
        out = self.predictor(out)

        if self.training:
            return out, new_lns, pred_log_len, true_log_len

        return out



