#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 15:46:43 2022

@author: atreyee
"""
import fairseq
import json
import librosa
import torch, torchaudio


audio_file, _ = librosa.load("19-198-0001.flac", sr=16000)
data_path = "./datasets/"
f = open('vocab.json')
target_dict = json.load(f)

model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(['./models/wav2vec_small.pt'], arg_overrides={'data': './'})
# model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(['./models/wav2vec_large.pt'])
model = model[0]
model.train()
# z = model.feature_extractor(torch.tensor(audio_file).unsqueeze(0))
audio_input = torch.tensor(audio_file).unsqueeze(0)
pad_mask = torch.zeros_like(audio_input)
# pad_mask[:,:10000] = 1
print(audio_input.shape)
x = model.extract_features(audio_input, pad_mask)
x_pad = model.extract_features(pad_mask, pad_mask)
print(x['padding_mask'].sum(1))

