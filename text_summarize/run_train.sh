#!/bin/bash

python ./SRC/train.py -mode train -encoder classifier -dropout 0.1 -bert_data_path ./bert_data/train/korean -model_path ./MODEL/KLUE/bert_classifier -lr 2e-3 -visible_gpus 0 -gpu_ranks 0 -world_size 1 -report_every 1000 -save_checkpoint_steps 500 -batch_size 1000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file ./LOG/KLUE/bert_classifier.txt -use_interval true -warmup_steps 200

# python ./SRC/train.py \
#   -mode train \
#   -encoder rnn \
#   -dropout 0.1 \
#   -bert_data_path ./bert_data/train/korean \
#   -model_path ./MODEL/KLUE/bert_rnn \
#   -lr 2e-3 \
#   -visible_gpus 0 \
#   -gpu_ranks 0 \
#   -world_size 1 \
#   -report_every 1000\
#   -save_checkpoint_steps 500 \
#   -batch_size 1000 \
#   -decay_method noam \
#   -train_steps 50000 \
#   -accum_count 2 \
#   -log_file ./LOG/KLUE/bert_rnn.txt \
#   -use_interval true \
#   -warmup_steps 200 \
#   -rnn_size 768

# python ./SRC/train.py \
#   -mode train \
#   -encoder transformer \
#   -dropout 0.1 \
#   -bert_data_path ./bert_data/train/korean \
#   -model_path ./MODEL/KLUE/bert_transformer \
#   -lr 2e-3 \
#   -visible_gpus 0 \
#   -gpu_ranks 0 \
#   -world_size 1 \
#   -report_every 1000\
#   -save_checkpoint_steps 500 \
#   -batch_size 1000 \
#   -decay_method noam \
#   -train_steps 50000 \
#   -accum_count 2 \
#   -log_file ./LOG/KLUE/bert_transformer.txt \
#   -use_interval true \
#   -warmup_steps 200 \
#   -ff_size 2048 \
#   -inter_layers 2 \
#   -heads 8