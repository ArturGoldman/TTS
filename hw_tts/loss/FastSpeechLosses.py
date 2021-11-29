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
        MSE = ((outputs-ground_truth_spectrogram)**2).sum()
        return MSE


class DPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, pred: Tensor, gt: Tensor):
        # pred: [batch_sz, lens]
        MSE = ((pred-gt)**2).sum()
        return MSE
