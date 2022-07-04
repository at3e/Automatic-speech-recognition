#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 17:55:49 2022

@author: atreyee
"""
import time
import io
import os
import string
import tempfile
import unittest

import torch
from fairseq import tokenizer
from fairseq.data import Dictionary

def add_file_to_dict(data_dir, dict_file_name):
    counts = {}
    num_lines = 100
    per_line = 10
   
    filename = os.path.join(data_dir, dict_file_name)
    with open(filename, "w", encoding="utf-8") as data:
        for c in string.ascii_letters:
            line = f"{c} " * per_line
            print(line)
            for _ in range(num_lines):
                data.write(f"{line}\n")
            counts[c] = per_line * num_lines
            per_line += 5

    dict = Dictionary()
    Dictionary.add_file_to_dictionary(
        filename, dict, tokenizer.tokenize_line, 10
    )
    dict.finalize(threshold=0, nwords=-1, padding_factor=8)

    return dict

def prepare_empty_dir(dirs, resume=False):
    """
    if resume experiment, assert the dirs exist,
    if not resume experiment, make dirs.

    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
        print(dir_path)
    """

    for dir_path in dirs:

        if resume:
            assert dir_path.exists()
        else:
            dir_path.mkdir(parents=True, exist_ok=True)

class ExecutionTime:
    """
    Usage:
        timer = ExecutionTime()
        <Something...>
        print(f"Finished in {timer.duration()} seconds.")
    """

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return time.time() - self.start_time
