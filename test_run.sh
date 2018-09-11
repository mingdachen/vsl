#!/bin/bash

source activate py35

python vsl_g.py \
    --debug 1 \
    --model g \
    --data_file twitter.data \
    --prior_file test_g \
    --vocab_file twitter \
    --tag_file tagger_twitter \
    --embed_file wordvects.tw100w5-m40-it2 \
    --n_iter 30000 \
    --save_prior 1 \
    --train_emb 0 \
    --tie_weights 1 \
    --embed_dim 100 \
    --latent_z_size 50 \
    --update_freq_label 1 \
    --update_freq_unlabel 1 \
    --rnn_size 100 \
    --char_embed_dim 50 \
    --char_hidden_size 100 \
    --mlp_layer 2 \
    --mlp_hidden_size 100 \
    --learning_rate 1e-3 \
    --vocab_size 100000 \
    --batch_size 10 \
    --kl_anneal_rate 1e-4 \
    --print_every 100 \
    --eval_every 1000 \
    --summarize 1
