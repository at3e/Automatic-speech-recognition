#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 16:42:37 2022

@author: atreyee
"""

import fairseq
import json
import librosa
import torch, torchaudio

import torch.multiprocessing as mp
import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp

from model import wav2vecModel
from datautil import AudioFileDataset
from torch.utils.data import DataLoader
from criterion import CtcCriterion
from train import Seq2SeqTraining

f = open('target_dict.json5')
tgt_dict = json.load(f)

# define dataloader
f = open("config.json5")
config = json.load(f)

# Create dataloader
data_folder = './'
train_dataset = AudioFileDataset(mode='train', device='cuda')
valid_dataset = AudioFileDataset(mode='validation', device='cuda')
# test_dataset = IRMDataset(mode='test', device=device)

batch_size = config['model'][0]['batch_size']
params = {"batch_size": 2, 
          "shuffle": True}
params_ = {"batch_size": 1, 
          "shuffle": True}
train_generator = DataLoader(train_dataset, **params)
val_generator = DataLoader(valid_dataset, **params_)
# load pre-trained checkpoint
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(['./models/wav2vec_small.pt'])
model = model[0]
asr_model = wav2vecModel(model, tgt_dict).to('cuda:1')
resume = False

# define train hyperparameters
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    
train_config = config["trainer"][0]
optimizer = torch.optim.Adam(model.parameters(), lr = train_config['lr'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose =True)
ctc_criterion = CtcCriterion(target_dictionary=tgt_dict)

trainer = Seq2SeqTraining(
    n_gpus = n_gpus,
    config = train_config,
    max_epoch = train_config['epochs'],
    resume_from_checkpoint = resume,
    criterion = ctc_criterion, 
    learning_rate = train_config['lr'],
    n_lr_warm_up_epoch = train_config['n_warm_up'],
    loss_coeff_dict = {},
    train_data_loader = train_generator,
    valid_data_loader = val_generator,
    model = asr_model
)
    
score = trainer.training_step()

