import torch
import torchaudio
import random
import numpy as np
import re
from unidecode import unidecode

# alignments and text preprocessing is taken from https://github.com/xcmyz/FastSpeech/blob/master/text/cleaners.py
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root, to_sr=22050, limit=None):
        super().__init__(root=root)
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
        cur_sz = super().__len__()
        self._index = list(range(cur_sz))
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

        transcript = english_cleaners(transcript)

        tokens, token_lengths = self._tokenizer(transcript)

        alignment = np.load('./alignments/' + str(self._index[index]) + '.npy')

        return waveform, waveform_length, transcript, tokens, token_lengths, alignment

    def __len__(self):
        return len(self._index)

    def decode(self, tokens, lengths):
        # not sure if this works
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result
