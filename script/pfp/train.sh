#! /bin/bash

BRANCH=('mf')
# BRANCH=('cc' 'mf' 'bp')

for BR in ${BRANCH[*]}; do
    python src/pfp/train.py \
        --branch ${BR} \
        --datadir ./data/processed/quickgo/ \
        --seqemb ./data/processed/emb/protein_seq \
        --struemb ./data/processed/emb/protein_stru \
        --goemb ./data/processed/emb/pretrained_go/${BR}/bert-base-uncased.pkl \
        --maxlen 800 \
        --batchmode term \
        --batchpro 16 \
        --shuffle \
        \
        --seqembsize 1024 \
        --struembsize 256 \
        --goembsize 256 \
        --hidsize 256 \
        --layers 6 \
        --dropout 0.2 \
        \
        --device 0 1 \
        --seed 100 \
        --savedir ./save/pfp/${ARCH} \
        --lr 3e-4 \
        --steps 100000 \
        --validinterval 1000 \
    
done