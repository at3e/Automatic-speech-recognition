#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:49:30 2022

@author: Atreyee
"""
"""
Implementation of flashlight beam search decoder
"""

import numpy as np
import math
from flashlight.lib.text.decoder import (
    CriterionType,
    LexiconDecoderOptions,
    KenLM,
    LexiconDecoder,
    SmearingMode,
    Trie,
)
from flashlight.lib.text.dictionary import (
    Dictionary,
    create_word_dict,
    load_words,
    pack_replabels,
)

import sys
sys.path.append("/idiap/group/speech/local/opt/")
import kenlm

beam_size = 2500
token_beam_size = 2500
beam_threshold = 100.0
lm_weight = 2.0
word_score = 2.0
unk_score = -math.inf
sil_score = -1
log_add = False
criterion_type = CriterionType.CTC

options = LexiconDecoderOptions(
        beam_size,
        token_beam_size,
        beam_threshold,
        lm_weight,
        word_score,
        unk_score,
        sil_score,
        log_add,
        criterion_type
        )

tokens_dict = Dictionary("./tokens.txt")

lexicon = load_words("./lexicon.txt")
word_dict = create_word_dict(lexicon)

lm = KenLM("./test.arpa", word_dict)

# test LM

# build trie

# get silence index
sil_idx = tokens_dict.get_index("|")
# get unknown word index
unk_idx = word_dict.get_index("<unk>")
# create the trie, specifying how many tokens we have and silence index
trie = Trie(tokens_dict.index_size(), sil_idx)
start_state = lm.start(False) # error!
for word, spellings in lexicon.items():
    usr_idx = word_dict.get_index(word)
    _, score = lm.score(start_state, usr_idx)
    for spelling in spellings:
        # max_reps should be 1; using 0 here to match DecoderTest bug
        spelling_idxs = tkn_to_idx(spelling, token_dict, 1)
        trie.insert(spelling_idxs, usr_idx, score)

trie.smear(SmearingMode.MAX)
