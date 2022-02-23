#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:27:24 2022

@author: atreyee
"""
import os
from operator import itemgetter
from dataclasses import dataclass, field
import torch
from typing import List
from queue import PriorityQueue
import kenlm


class BeamSearchNode(object):
    def __init__(self, lprob, previousNode, ltrId, length):
        
        self.prevNode = previousNode
        self.ltrid = ltrId
        self.lprob = lprob
        self.len = length

    def eval(self, alpha=1.0):
        reward = 0
       
        return self.lprob / float(self.len - 1 + 1e-6) + alpha * reward


class BeamSearchDecoder(torch.nn.Module):
    def __init__(self, tgt_dict, blank=0):
        super().__init__()
        self.labels = self.get_labels(tgt_dict)
        self.tgt_dict = tgt_dict
        self.blank = blank
        self.SOS_token = tgt_dict['<s>']
        self.EOS_token = tgt_dict['</s>']
        self.beam_width = 5
        self.nsent = 3  # number of sentences to generate
        
        
    def get_labels(self, tgt_dict):
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

    def forward(self, predicted_seq_lprobs: torch.Tensor, target_seq: torch.Tensor, seq_len: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get topk best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: self.nsent resulting transcript
        """
        
        decoded_batch = []
        nodes = []
            
        EOSnodes = []
        lprob_idx = predicted_seq_lprobs[:, 0, 0]
                   
        # Start with the <s> of the sentence token
        decoder_input = torch.LongTensor([[self.SOS_token]])
        
        SOSnode = BeamSearchNode(lprob_idx, None, self.tgt_dict['<s>'], 1)
        EOSnode = None
        # Start queue
        nodes.append([(-SOSnode.eval(), SOSnode)])
        qsize = 1

        # iterate over sequence
        for idx in range(1, target_seq.size(-1)):
            
            # Start search
            
            # give up when decoding takes too long
            if qsize > 2000: break
            # fetch next step
            lprob_next_idx = predicted_seq_lprobs[:, idx, :].unsqueeze(0)
            lprob_, indexes = torch.topk(lprob_next_idx, self.beam_width)
            
            nextnodes = [] 
            
            for n_ in nodes[-1]:
                score, n = n_
        
                if n.ltrid == self.EOS_token and n.prevNode != None:
                    EOSnodes.append((-n.eval(), n))
                    continue

                for k in range(self.beam_width):
                    topk_index = indexes[:, 0, k].view(1, -1)
                    lprob = lprob_next_idx[:, 0, k].item()

                    node = BeamSearchNode(n.lprob + lprob, n, self.tgt_dict[self.labels[topk_index]], n.len + 1)
                    score = -node.eval().item()
                    nextnodes.append((score, node))
                    
            nextnodes.sort(key=itemgetter(0))
            nodes.append(nextnodes[:self.nsent])
             
        # back trace best paths
        if len(EOSnodes) == 0:
            EOSnodes = [nextnodes[i] for i in range(self.nsent)]

        # decoded_sentences = []
        decoded_toks = []

        for score, n in sorted(EOSnodes, key=lambda s: s[0]):
            id_seq = []
            id_seq.append(n.ltrid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                id_seq.append(n.ltrid)
            id_seq = id_seq[::-1]
            
            decoded_toks.append(id_seq)
            # # Convert to letters
            # ids = torch.unique_consecutive(torch.tensor(id_seq), dim=-1)
            # ids = [i for i in ids if i != self.blank]
            # decoded_ = "".join([self.labels[i] for i in ids])
            # decoded_.replace("|", " ").strip().split()
            # decoded_sentences.append(decoded_.lower())


        return decoded_toks 
