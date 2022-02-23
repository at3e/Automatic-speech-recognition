#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 00:38:20 2022

@author: root
"""
import os
import json
import pandas as pd
import librosa


import torch
from torch.utils.data import Dataset
import torch.nn.functional as F




class AudioFileDataset(Dataset):
    def __init__(self,
                 train_f_paths = '/home/atreyee/ASR/datasets/labels_train-clean-100/dev_other_.tsv',
                 valid_f_paths = '/home/atreyee/ASR/datasets/labels_train-clean-100/dev_other_.tsv',
                 train_ltr_path = '/home/atreyee/ASR/datasets/labels_train-clean-100/dev_other_.ltr',
                 valid_ltr_path = '/home/atreyee/ASR/datasets/labels_train-clean-100/dev_other_.ltr',
                 vocab_file = 'target_dict.json5',
                 fs = 16000,
                 max_len = 10,
                 max_token_len = 200,
                 mode = "train",
                 device = 'cpu'):

        super().__init__()
        
        self.device = device  
        self.root = '/home/atreyee/ASR/datasets/dev-other'
        self.train_files = pd.read_csv(train_f_paths)
        self.valid_files = pd.read_csv(valid_f_paths)
        
        self.mode = mode
        
        #Read labels
        f = open(train_ltr_path)
        self.train_labels = f.readlines()
        f.close()
        f = open(valid_ltr_path)
        self.valid_labels = f.readlines()
        f.close()
        
        f = open(vocab_file)
        self.target_dict = json.load(f)
        f.close()
        
        self.fs = fs
        self.max_len = max_len*self.fs
        self.max_token_len = max_token_len
        
        if mode=="train":
            self.length = len(self.train_files)
        elif mode=="validation":
            self.length = len(self.valid_files)
      
       
    def __len__(self):
        return self.length
    
    def make_pad_mask(self, length, xs=None, length_dim=-1):
    
        if length_dim == 0:
            raise ValueError('length_dim cannot be 0: {}'.format(length_dim))

    
        seq_range = torch.arange(0, self.max_len, dtype=torch.int64)
        seq_range_expand = seq_range.unsqueeze(0).expand(1, self.max_len)
        seq_length_expand = seq_range_expand.new(length).unsqueeze(-1)
        mask = seq_range_expand >= seq_length_expand
    
        if xs is not None:
            assert xs.size(0) == 1, (xs.size(0), 1)
    
            if length_dim < 0:
                length_dim = xs.dim() + length_dim
            
            ind = tuple(slice(None) if i in (0, length_dim) else None
                        for i in range(xs.dim()))
            mask = mask[ind].expand_as(xs).to(xs.device)
        return mask

    def __getitem__(self, idx):
        sample = {}

        if self.mode == "train":
            filename = self.train_files.iloc[idx,0].split('\t')
            wav_input, _ = librosa.load(os.path.join(self.root, filename[0]))
            wav_len = len(wav_input)
            
            trans = self.train_labels[idx].strip().split(" ")
            tokens = [self.target_dict['<s>']]
            for ltr in trans:
                if ltr in self.target_dict.keys():
                    tokens.append(self.target_dict[ltr])
            tokens.append(self.target_dict['</s>'])
            tokens = torch.tensor(tokens).to('cuda:1')
            
            if wav_len>self.max_len:
                padding_mask = torch.zeros(self.max_len)
                wav_input = torch.torch.tensor(wav_input[:self.max_len])
                
            else:
                padding_mask = F.pad(torch.zeros(len(wav_input)),
                                      (0,self.max_len-wav_len), mode='constant', value=1)                
                wav_input = F.pad(torch.torch.tensor(wav_input), 
                                  (0,self.max_len-wav_len), mode='constant', value=0)
                
            if len(tokens)>self.max_token_len:
                tokens = tokens[:self.max_token_len]
            else:
                tokens = F.pad(tokens, (0,self.max_token_len-len(tokens)),
                               mode='constant', value=self.target_dict['<pad>'])                          
               
            sample['net_input'] = wav_input.to('cuda:1')
            sample['padding_mask'] = padding_mask.to('cuda:1')
            sample['src_lengths'] = wav_len
            sample['target'] = torch.tensor(tokens).to('cuda:1')
            sample['ntokens'] = len(tokens)-1
            sample['id'] = idx
            
            return sample

        elif self.mode == "validation":
            filename = self.valid_files.iloc[idx,0].split('\t')
            wav_input, _ = librosa.load(os.path.join(self.root, filename[0]))
            wav_input = torch.torch.tensor(wav_input)
            wav_len = len(wav_input)
            padding_mask = torch.zeros_like(wav_input)
            trans = self.valid_labels[idx].strip().split(" ")
            
            tokens = [self.target_dict['<s>']]
            for ltr in trans:
                if ltr in self.target_dict.keys():
                    tokens.append(self.target_dict[ltr])
            tokens.append(self.target_dict['</s>'])
            tokens = torch.tensor(tokens).to('cuda:1')# load to device in the 
            
            if wav_len>self.max_len:
                padding_mask = torch.zeros(self.max_len)
                wav_input = torch.torch.tensor(wav_input[:self.max_len])
                
            else:
                padding_mask = F.pad(torch.zeros(len(wav_input)),
                                      (0,self.max_len-wav_len), mode='constant', value=1)                
                wav_input = F.pad(torch.torch.tensor(wav_input), 
                                  (0,self.max_len-wav_len), mode='constant', value=0)
                
            if len(tokens)>self.max_token_len:
                tokens = tokens[:self.max_token_len]
            else:
                tokens = F.pad(tokens, (0,self.max_token_len-len(tokens)),
                               mode='constant', value=self.target_dict['<pad>'])
            
            sample['net_input'] = wav_input.to('cuda:1')
            sample['padding_mask'] = padding_mask.to('cuda:1')
            sample['src_lengths'] = wav_len
            sample['target'] = torch.tensor(tokens).to('cuda:1')
            sample['ntokens'] = len(tokens)-1
            sample['id'] = idx
            
            return sample


def sentence_post_process(sentence, symbol):
    
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == 'wordpiece':
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == 'letter':
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol is not None and symbol != 'none':
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    return sentence
