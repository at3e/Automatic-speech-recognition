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


model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(['./wav2vec_small_960h.pt'], \
                                                                          arg_overrides={"data":"/home/atreyee/ASR/datasets/labels_train-clean-100/"})
# model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(['./models/wav2vec_large.pt'])
model = model[0]
model.train()
# z = model.feature_extractor(torch.tensor(audio_file).unsqueeze(0))
z = model.w2v_encoder(torch.tensor(audio_file).unsqueeze(0))
x = model.extract_features(z.squeeze(0))
dev_clean_librispeech_data = torchaudio.datasets.LIBRISPEECH(data_path, url='dev-clean', download=False)
data_loader = torch.utils.data.DataLoader(dev_clean_librispeech_data, batch_size=1, shuffle=False)
