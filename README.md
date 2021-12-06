# TTS
Text To Speech HW for DLA HSE course

## Running guide

All commands are written as if they are executed in Google Colab

To set up environment run
```
!git clone -b overfit https://github.com/ArturGoldman/TTS
!chmod u+x ./TTS/prep.sh
!./TTS/prep.sh
```

To start testing run
```
!chmod u+x ./TTS/test.sh
! ./TTS/test.sh
```

By default training sentences are specified in `sentences.txt`. If you want to pass you own file with sentences,
place it in the same directory and provide path to it in `config_test.json` under 'file_dir' field. Then, execute commands above.

If you want to start training process from the start run
```
!python3 ./TTS/train.py -c ./TTS/hw_tts/config.json
```
Note that after training you will have to pass trained model to test on your own. See `test.sh`.

## Results
~~ Under construction ~~

## Implementation details
~~ Under construction ~~

## Credits
Some transformer implementation details are based on 
[pytorch tutorial and official pytorch implementation](https://pytorch.org/tutorials/beginner/translation_transformer.html).

Multihead-Attention is built as described in [this illustrated Transformer tutorial](https://jalammar.github.io/illustrated-transformer/).

Structure of this repository is based on [template repository of first ASR homework](https://github.com/WrathOfGrapes/asr_project_template),
which itself is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.
