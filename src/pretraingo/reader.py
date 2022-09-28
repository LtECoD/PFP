import os
import torch
import pickle

from embedder import Embedder


def read_graph(args):
    graphdir, model_name, embdir = args.godir, args.lm, args.embdir
    os.makedirs(embdir, exist_ok=True)

    n2idx_name = "term_index.tsv"
    nn_desc = "term_desc.tsv"
    adj_name = "adj.tsv"

    with open(os.path.join(graphdir, n2idx_name), "r") as f:
        n2idx = {}
        idx2n = {}
        for l in f.readlines()[1:]:
            n, stri = l.strip().split("\t")
            n2idx[n] = int(stri)
            idx2n[int(stri)] = n
        assert len(n2idx) == len(idx2n)
    
    with open(os.path.join(graphdir, adj_name), "r") as f:
        srcs, etypes, tgts = [], [], []
        for l in f.readlines()[1:]:
            s, e, t = l.strip().split("\t")[3:]
            srcs.append(int(s))
            etypes.append(int(e))
            tgts.append(int(t))

    """get initial embeddings"""
    ini_embed_fp = os.path.join(embdir, model_name+".pkl")
    if os.path.exists(ini_embed_fp):
        with open(ini_embed_fp, "rb") as f:
            embs = pickle.load(f)
    else:
        with open(os.path.join(graphdir, nn_desc), "r") as f:
            n2d = {}
            for l in f.readlines()[1:]:
                _, idx, desc = l.strip().split("\t")
                n2d[int(idx)] = desc
            assert len(n2d) == len(n2idx)
        # embedding
        embedder = Embedder(model_name=model_name)
        embs = embedder.forward(n2d)
        with open(os.path.join(embdir, model_name+".pkl"), "wb") as f:
            pickle.dump(embs, f)
            
    # initial embes into tensors in index order
    feat = []
    for idx in range(len(embs)):
        feat.append(torch.Tensor(embs[idx]))
    assert len(feat) == len(idx2n)
    return srcs, etypes, tgts, feat