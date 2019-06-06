#!/usr/bin/env sh
python -u train_align.py \
    --batch_size 512 \
    --part_npoints 1024 \
    --dense_npoints 1024 \
    --max_batch 10000000000 \
    --learning_rate 0.1 \
    --decay_step 1024000\
    --decay_rate 0.5 \
    --model align_model \
    --snapshot 1000 \
    --log_dir ./checkpoint/ \
    --display_iter 25 \
    --restore \
    --log_file ./checkpoint/model_best.ckpt


   