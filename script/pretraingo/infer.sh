#! /bin/bash

python src/pretraingo/infer.py \
    --lm bert-base-uncased \
    --godir data/processed/gograph \
    --savedir save/pretraingo \
    --embdir data/processed/emb/initialized_go \
    --ptrdir data/processed/emb/pretrained_go \
    \
    --hid_dim 512 \
    --out_dim 256 \
    --dropout 0.1 \
    --device 1