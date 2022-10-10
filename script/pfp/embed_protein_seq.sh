python src/pfp/embed_protein_seq.py  \
    --pretrained_model Rostlab/prot_t5_xl_uniref50 \
    --seqfile data/processed/quickgo/seqs.tsv \
    --embdir data/processed/emb/protein_seq \
    --device 0 \
    --batch_size 32