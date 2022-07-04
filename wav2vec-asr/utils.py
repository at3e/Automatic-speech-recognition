#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 17:55:49 2022

@author: atreyee
"""
import time

def test_add_file_to_dict(self):
        counts = {}
        num_lines = 100
        per_line = 10
        with tempfile.TemporaryDirectory("test_sampling") as data_dir:
            filename = os.path.join("dict.ltr.txt")
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

            for c in string.ascii_letters:
                count = dict.get_count(dict.index(c))
                self.assertEqual(
                    counts[c], count, f"{c} count is {count} but should be {counts[c]}"
                )

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
