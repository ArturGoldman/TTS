import torch
from torch import nn, Tensor
from hw_tts.aligner import Batch
from hw_tts.processing import MelSpectrogram


class FTLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.melspec = MelSpectrogram(config)

    def __call__(self, outputs: Tensor, batch: Batch):
        # outputs: [batch_sz, seq_len, n_mels]
        ground_truth_spectrogram = self.melspec(batch.waveform)
        MSE = ((outputs-ground_truth_spectrogram)**2).mean()
        return MSE


class DPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, batch: Batch, pred: Tensor, gt: Tensor):
        # pred: [batch_sz, lens], lens are not compatible with gt
        MSE = 0
        for i in range(pred.size(0)):
            MSE += ((pred[i, :batch.token_lengths[i]]-torch.log(gt[i]))**2).mean()
        return MSE/pred.size(0)
