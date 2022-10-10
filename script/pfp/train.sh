#! /bin/bash

ARCH=lstm
CRITEION=cross_entropy

BRANCH=('bp')
# BRANCH=('cc' 'mf' 'bp')
for BR in ${BRANCH[*]}; do

    fairseq-train \
        --user-dir src/pfp \
        --save-dir ./save/pfp/${ARCH} \
        --seed 100 \
        \
        --optimizer adam \
        --lr 3e-4 \
        --batch-size 16 \
        --max-epoch 5 \
        \
        --branch ${BR} \
        --datadir ./data/processed/quickgo/ \
        --seqembdir ./data/processed/emb/protein_seq \
        --struembdir ./data/processed/emb/protein_stru \
        --goembfile ./data/processed/emb/pretrained_go/${BR}/bert-base-uncased.pkl \
        --maxlen 800 \
        --train-subset train \
        --valid-subset test \
        \
        --task pfp \
        --arch ${ARCH} \
        # # --criterion ${CRITEION} \
        # # # \
        # # # --dropout 0.2 \
        # # # --emb-dim 1024 \
        # # # --hid-dim 256 \
        # # # --trans-layers 8

done