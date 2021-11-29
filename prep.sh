#!/bin/bash

pip install -r ./TTS/requirements.txt

pip install torch==1.10.0+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2

git clone https://github.com/NVIDIA/waveglow.git
pip install googledrivedownloader