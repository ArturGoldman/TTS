# TTS
Text To Speech HW for DLA HSE course

This `overfit branch` contains version, which successfully overfits on batch to show correctness of created model.

---
To start overfitting in Colab run
```
!git clone -b overfit https://github.com/ArturGoldman/TTS
!chmod u+x ./TTS/prep.sh
!./TTS/prep.sh
!python ./TTS/train.py -c ./TTS/hw_tts/config.json
```
