import torch
import torchaudio
import random
from hw_tts.aligner import GraphemeAligner
from tqdm import tqdm


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root, to_sr=22050, limit=None):
        super().__init__(root=root)
        self.aligner = GraphemeAligner()
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
        self._index = []
        cur_sz = super().__len__()
        for i in tqdm(range(cur_sz), desc="Dataset Filtering", total=cur_sz):
            _, _, _, transcript = super().__getitem__(i)
            tokens, token_lengths = self._tokenizer(transcript)
            tokens_other = self.aligner._decode_text(transcript)
            if tokens.size(-1) == tokens_other.size(-1):
                self._index.append(i)
        print("Old len: {}, new len: {}".format(cur_sz, len(self._index)))
        self.limit = limit
        self.to_sr = to_sr
        random.seed(42)
        random.shuffle(self._index)
        if limit is not None:
            self._index = self._index[:limit]

    def __getitem__(self, index: int):
        waveform, old_sr, _, transcript = super().__getitem__(self._index[index])
        waveform = torchaudio.transforms.Resample(old_sr, self.to_sr)(waveform)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        tokens, token_lengths = self._tokenizer(transcript)

        return waveform, waveform_length, transcript, tokens, token_lengths

    def __len__(self):
        return len(self._index)

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result
