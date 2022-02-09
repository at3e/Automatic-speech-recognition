#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:27:24 2022

@author: atreyee
"""
import os
import torch
from typing import List
from queue import PriorityQueue
import kenlm


"""
    Beam search decoder adapted from:
        https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
"""

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
       
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
    
    
class BeamSearchDecoder(torch.nn.Module):
    def __init__(self, tgt_dict, blank=0):
        super().__init__()
        self.labels = self.create_labels(tgt_dict)
        self.blank = blank
        self.SOS_token = tgt_dict['<s>']
        self.EOS_token = tgt_dict['</s>']
        self.beam_width = 10
        self.topk = 3  # number of sentences to generate
        self.LM = os.path.join('/home/atreyee/ASR','test.arpa')
        self.model = kenlm.LanguageModel(self.LM)
        
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

    def forward(self, emission: torch.Tensor, decoded: torch.Tensor, target_seq: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get topk best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: self.topk resulting transcript
        """
        
        decoded_batch = []
        
        # decode by seq
        for idx in range(target_seq.size(0)):
            decoder_hid = decoded[:, idx, :].unsqueeze(0)
            emission_ = emission[:,idx, :].unsqueeze(1)
           
            # Start with the <s> of the sentence token
            decoder_input = torch.LongTensor([[self.SOS_token]])
            
            # Num sentences to generate
            endnodes = []
            num = min((self.topk + 1), self.topk - len(endnodes))
            
            node = BeamSearchNode(decoder_hid, None, decoder_input, 0, 1)
            nodes = PriorityQueue()
            
            # Start queue
            nodes.put((-node.eval(), node))
            qsize = 1
            
            # Start search
            
            while True:
                # give up when decoding takes too long
                if qsize > 2000: break
    
                # fetch the best node
                score, n = nodes.get()
                decoder_input = n.wordid
                decoder_hid = n.h
                
                if n.wordid.item() == self.EOS_token and n.prevNode != None:
                    endnodes.append((score, n))
                    
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= num:
                        break
                    else:
                        continue
                    
                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoded, self.beam_width)
                nextnodes = []
    
                for new_k in range(self.beam_width):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()
    
                    node = BeamSearchNode(decoder_hid, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))
                    
                # insert into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1
                
                # back trace nbest paths
                if len(endnodes) == 0:
                    endnodes = [nodes.get() for _ in range(self.topk)]
                    
            sum_inv_logprob = []    
            for ind, sentence in enumerate(endnodes):
                sum_inv_logprob[ind] = -1.0 * sum(score for score, _, _ in self.model.full_scores(sentence))
        
        indices = endnodes[endnodes.index(max(endnodes))]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])
        
        return joined.replace("|", " ").strip().split()