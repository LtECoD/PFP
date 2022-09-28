#! /bin/bash

CATS=('cc' 'mf' 'bp')

for CAT in ${CATS[*]}; do

    python src/pretraingo/train.py \
        --lm bert-base-uncased \
        --godir data/processed/gograph/${CAT} \
        --embdir data/processed/emb/initialized_go/${CAT} \
        --savedir save/pretraingo/${CAT} \
        --resultdir result/pretraingo/${CAT} \
        \
        --hid_dim 512 \
        --out_dim 256 \
        --dropout 0.1 \
        \
        --seed 99 \
        --device 1 \
        --lr 1e-3 \
        --epoc 1500 \
        --nmask_ratio 0.2 \
        --emask_ratio 0.2

done

