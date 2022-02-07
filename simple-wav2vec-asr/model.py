#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 15:11:08 2022

@author: atreyee
"""

import torch
import torch.nn as nn



class wav2vecModel(nn.Module):
    def __init__(self, encoder, tgt_dict):
        super(wav2vecModel, self).__init__()
        self.d = 768
        self.encoder = encoder
        if tgt_dict is not None:
            self.proj = nn.Linear(self.d, len(tgt_dict))

    def forward(self, net_input, padding_mask, target, ntokens, id):
        with torch.no_grad():
            emissions, mask = self.encoder.extract_features(net_input, padding_mask)
        lprobs = self.encoder.get_normalized_probs(emissions, log_probs=True).contiguous()  # (T, B, C) from the encoder
        net_output = self.proj(emissions.transpose(1,0))
        return net_output, mask