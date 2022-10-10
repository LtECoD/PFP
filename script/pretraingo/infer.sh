#! /bin/bash

CATS=('cc' 'mf' 'bp')

for CAT in ${CATS[*]}; do
    python src/pretraingo/infer.py \
        --lm bert-base-uncased \
        --godir data/processed/gograph/${CAT} \
        --savedir save/pretraingo/${CAT} \
        --embdir data/processed/emb/initialized_go/${CAT} \
        --ptrdir data/processed/emb/pretrained_go/${CAT} \
        \
        --hid_dim 512 \
        --out_dim 256 \
        --dropout 0.1 \
        --device 1
done