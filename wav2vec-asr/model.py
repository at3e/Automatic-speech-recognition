#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 15:11:08 2022

@author: atreyee
"""

import torch
import torch.nn as nn



class wav2vecModel(nn.Module):
    def __init__(self, encoder, decoder, tgt_dict):
        super(wav2vecModel, self).__init__()
        self.d = 768
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, net_input, padding_mask, src_lengths, target, ntokens, id):
        with torch.no_grad():
            emissions, mask = self.encoder.extract_features(net_input, padding_mask)
        net_output, pad_mask = self.decoder(emissions.transpose(1,2), src_lengths)
        return net_output.permute(2, 0, 1), pad_mask