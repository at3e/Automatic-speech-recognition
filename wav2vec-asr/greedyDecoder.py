#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 03:08:18 2022

@author: atreyee
"""

from typing import List
import json
import torch
import torch.nn as nn

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, tgt_dict, blank=0):
        super().__init__()
        self.labels = self.create_labels(tgt_dict)
        self.blank = blank
        
    def create_labels(self, tgt_dict):
        labels = [0]*len(tgt_dict)
        for key, value in tgt_dict.items():
            labels[value] = key
            
        return labels
    
    def to_string(self, labels):
        indices = torch.unique_consecutive(labels, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])
        return joined #.replace("|", " ").strip().split()

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])
        return joined.replace("|", " ").strip().split()
    


# # tokens = ['-', '|', 'e', 't', 'a', 'o', 'n', 'i', 'h', 's', 'r', 'd', 'l', 'u', 'm', 'w', 'c', 'f', 'g', 'y', 'p', 'b', 'v', 'k', "'", 'x', 'j', 'q', 'z']
# f = open('target_dict.json5')
# tgt_dict = json.load(f)
# greedy_decoder = GreedyCTCDecoder(tgt_dict)
# lprobs = torch.randn(67, 1, 29)
# decoded = greedy_decoder.forward(lprobs)