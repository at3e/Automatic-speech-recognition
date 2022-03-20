if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Enter stage 1"
    echo "Prepare manifest TSV files"
    tempdir=$(mktemp -d)
    python ${fairseq_root}/examples/wav2vec/wav2vec_manifest.py \
        ${lang_dir}/${train_set}/wavs \
        --dest "${tempdir}" \
        --ext $wav_ext --valid-percent 0
    mv "${tempdir}/train.tsv" ${lang_dir}/${train_set}/train.tsv
    
    python ${fairseq_root}/examples/wav2vec/wav2vec_manifest.py \
        ${lang_dir}/${dev_set}/wavs \
        --dest "${tempdir}" \
        --ext $wav_ext --valid-percent 1.
    mv "${tempdir}/valid.tsv" ${lang_dir}/${dev_set}/valid.tsv
    
    rm -r "${tempdir}"
    cp ${lang_dir}/${dev_set}/valid.tsv  ${lang_dir}/${dev_set}/valid.tsv.orig
    cp ${lang_dir}/${train_set}/train.tsv  ${lang_dir}/${train_set}/train.tsv.orig

    echo "Exit stage 1"
fi


if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Enter stage 2"
    echo "Filter long sentences and prepare training labels"
    echo "Only supporting characters at the moment"

    echo "Filter long sentences"
    cp ${lang_dir}/${train_set}/train.tsv.orig \
        ${lang_dir}/${train_set}/train.tsv
    cp ${lang_dir}/${dev_set}/valid.tsv.orig \
        ${lang_dir}/${dev_set}/valid.tsv

    python filter_long_sentences.py \
      ${lang_dir}/${train_set}/train.tsv \
      --threshold $max_tokens \
      --output-dir ${lang_dir}/${train_set} \
      --output-name train

    python filter_long_sentences.py \
      ${lang_dir}/${dev_set}/valid.tsv \
      --threshold $max_tokens \
      --output-dir ${lang_dir}/${dev_set} \
      --output-name valid

    mv ${lang_dir}/${dev_set}/valid.filtered.tsv \
        ${lang_dir}/${dev_set}/valid.tsv

    mv ${lang_dir}/${train_set}/train.filtered.tsv \
        ${lang_dir}/${train_set}/train.tsv \

    echo "Prepare labels"
    python text_to_letters.py \
        ${lang_dir}/${train_set}/train.tsv \
        --trans-file ${lang_dir}/${train_set}/text \
        --nosymb-file ${lang_dir}/${nlsyms_txt} \
        --output-dir ${lang_dir}/${train_set} \
        --output-name train

    python text_to_letters.py \
        ${lang_dir}/${dev_set}/valid.tsv \
        --trans-file ${lang_dir}/${dev_set}/text \
        --nosymb-file ${lang_dir}/${nlsyms_txt} \
        --output-dir ${lang_dir}/${dev_set} \
        --output-name valid

    mkdir -p data-bin; rm -rf data-bin

    sed -i 's/<unk>/<oov>/g' ${lang_dir}/${train_set}/train.ltr
    sed -i 's/<unk>/<oov>/g' ${lang_dir}/${train_set}/train.wrd
    sed -i 's/<unk>/<oov>/g' ${lang_dir}/${dev_set}/valid.wrd
    sed -i 's/<unk>/<oov>/g' ${lang_dir}/${dev_set}/valid.ltr

    python ${fairseq_root}/fairseq_cli/preprocess.py \
      --dataset-impl mmap \
      --trainpref ${lang_dir}/${train_set}/train.ltr \
      --only-source \
      --thresholdsrc 0

    mv data-bin/dict.txt ${lang_dir}/${train_set}/dict.ltr.txt

    echo "Exit stage 2"
fi
