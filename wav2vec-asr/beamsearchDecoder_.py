#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:27:24 2022

@author: atreyee
"""
import os
from dataclasses import dataclass, field
import torch
from typing import List
from queue import PriorityQueue
import kenlm


"""
    Beam search decoder adapted from:
        https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
"""

class BeamSearchNode(object):
    def __init__(self, lprob, previousNode, ltrId, length):
        
        self.prevNode = previousNode
        self.ltrid = ltrId
        self.lprob = lprob
        self.len = length

    def eval(self, alpha=1.0):
        reward = 0
       
        return self.lprob / float(self.len - 1 + 1e-6) + alpha * reward

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: object = field()
    
class BeamSearchDecoder(torch.nn.Module):
    def __init__(self, tgt_dict, blank=0):
        super().__init__()
        self.labels = self.create_labels(tgt_dict)
        self.tgt_dict = tgt_dict
        self.blank = blank
        self.SOS_token = tgt_dict['<s>']
        self.EOS_token = tgt_dict['</s>']
        self.beam_width = 5
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

    def forward(self, predicted_seq_lprobs: torch.Tensor, target_seq: torch.Tensor, seq_len: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get topk best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: self.topk resulting transcript
        """
        
        decoded_batch = []
        
        # iterate over sequence
        for idx in range(target_seq.size(-1)):
            lprob_idx = predicted_seq_lprobs[:, idx, 0].item()
                       
            # Start with the <s> of the sentence token
            decoder_input = torch.LongTensor([[self.SOS_token]])
            
            # Num sentences to generate
            EOSnodes = []
            num = min((self.topk + 1), self.topk - len(EOSnodes))
            
            node = BeamSearchNode(lprob_idx, None, self.tgt_dict['<s>'], 1)
            nodes = PriorityQueue()
            
            # Start queue
            nodes.put(PrioritizedItem(-node.eval(), node))
            qsize = 1

            # Start search
            t = 1
            
            for t in range(1, seq_len):
                # give up when decoding takes too long
                if qsize > 2000: break

                n_ = nodes.get(False)
                score, n = n_.priority, n_.item
            
                if n.ltrid == self.EOS_token and n.prevNode != None:
                    EOSnodes.append((score, n))
                
                    # track num sentences recorded
                    if len(EOSnodes) >= num:
                        break
                    else:
                        continue

                # fetch next step
                lprob_next_idx = predicted_seq_lprobs[:, t, :].unsqueeze(0)
                lprob_, indexes = torch.topk(lprob_next_idx, self.beam_width)
                nextnodes = []

                for k in range(self.beam_width):
                    topk_index = indexes[:, 0, k].view(1, -1)
                    lprob = lprob_next_idx[:, 0, k].item()

                    node = BeamSearchNode(n.lprob + lprob, n, self.tgt_dict[self.labels[topk_index]], n.len + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # insert into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put(PrioritizedItem(score, nn))

                # increase qsize
                qsize += len(nextnodes) - 1

                # back trace best paths
                if len(EOSnodes) == 0:
                    EOSnodes = [nodes.get(False) for _ in range(self.topk)]

                decoded_sentence = []
                for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                    utterance = []
                    utterance.append(n.wordid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)

            sum_inv_lprob = []
            for i, sentence in enumerate(EOSnodes):
                sum_inv_lprob[i] = -1.0 * sum(score for score, _, _ in self.LMmodel.full_scores(sentence))

        indices = endnodes[endnodes.index(max(endnodes))]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])

        return joined.replace("|", " ").strip().split()