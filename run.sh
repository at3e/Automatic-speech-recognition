#!/usr/bin/bash

set -e
set -u
set -o pipefail

eval $(/idiap/group/speech/local/bin/brew shellenv)
kenlm_root = /idiap/group/speech/local/opt/kenlm

stage=1
wax_ext="flac"


train_set = 
dev_set =
eval_set =


max_tokens=10000
lang_dir=dataset/

