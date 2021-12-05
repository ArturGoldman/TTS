import torch
from typing import Tuple, Dict, Optional, List, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class Batch:
    waveform: torch.Tensor
    waveform_length: torch.Tensor
    transcript: List[str]
    tokens: torch.Tensor
    token_lengths: torch.Tensor
    alignment: List[np.ndarray]
    durations: Optional[torch.Tensor] = None
    pad_value: float = -11.5129251

    def to(self, device: torch.device) -> 'Batch':
        self.waveform = self.waveform.to(device)
        self.waveform_length = self.waveform_length.to(device)
        self.tokens = self.tokens.to(device)
        self.token_lengths = self.token_lengths.to(device)
        if self.durations is not None:
            self.durations = self.durations.to(device)
