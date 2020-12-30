#!/bin/bash

DATA_DIR='../../data/t-ner/'
MODEL_TYPE='bert'
MODEL_NAME_OR_PATH='hfl/chinese-roberta-wwm-ext-large'
OUTPUT_DIR='../bert-crf-model'
LABEL='../../data/labels.txt'
CACHE='../cache'
CUDA_VISIBLE_DEVICES='0' python ../examples/run_crf_ner.py \
    --data_dir $DATA_DIR \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR \
    --labels $LABEL \
    --cache_dir $CACHE \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --adv_training fgm \
    --num_train_epochs 100 \
    --max_seq_length 512 \
    --logging_steps 0.5 \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --bert_lr 1e-5 \
    --classifier_lr 1e-5 \
    --crf_lr 5e-3 \
