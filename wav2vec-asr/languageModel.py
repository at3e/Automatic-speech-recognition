#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:47:35 2022

@author: atreyee
"""
import os
import torch 
import kenlm
import json


class LanguageModel(torch.nn.Module):
    def __init__(self, tgt_dict, blank=0):
        self.LM = os.path.join('/home/atreyee/ASR','test.arpa')
        self.model = kenlm.LanguageModel(self.LM)
        self.blank = blank
        self.labels = self.create_labels(tgt_dict)
        
    def create_labels(self, tgt_dict):
        labels = [0]*len(tgt_dict)
        for key, value in tgt_dict.items():
            labels[value] = key
            
        return labels
        
    def to_string(self, id_seq):
        ids = torch.unique_consecutive(torch.tensor(id_seq), dim=-1)
        ids = [i for i in ids if i != self.blank]
        decoded_ = "".join([self.labels[i] for i in ids])
        decoded_.replace("|", " ").strip().split()
        return decoded_.lower()
        
    def LMscore(self, decoded_tok):
        sum_inv_logprob = []
        for tok in decoded_tok:
            sentence = self.to_string(tok)
            sum_inv_logprob.append(-1.0 * sum(score for score, _, _ in self.model.full_scores(sentence)))
        # print(sum_inv_logprob)
            
        return sum_inv_logprob
 
# f = open('target_dict.json5')
# tgt_dict = json.load(f)       
# tokens = [[0, 2, 3, 7, 0, 9], [0, 2, 3, 7, 0, 9]]
# LangMod = LanguageModel()
# _ = LangMod.LMscore(tokens)