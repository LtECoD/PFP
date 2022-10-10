import os
import dgl
import torch
import pickle
import argparse

from model import Model
from reader import read_graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", type=str, help="language model")
    parser.add_argument("--godir", type=str, help="input go graph directory")
    parser.add_argument("--embdir", type=str, help="dir to store initial embedding")
    parser.add_argument("--ptrdir", type=str, help="dir to store pretrained embedding")
    parser.add_argument("--savedir", type=str)

    # model parameters
    parser.add_argument("--hid_dim", type=int, help="dimension of hidden")
    parser.add_argument("--out_dim", type=int, help="output dimension")
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    os.makedirs(args.ptrdir, exist_ok=True)

    srcs, etypes, tgts, feat = read_graph(args)
    graph = dgl.graph((srcs, tgts), num_nodes=len(feat), device=args.device)
    feat = torch.stack(feat, dim=0).to(args.device)
    etypes = torch.LongTensor(etypes).to(args.device)

    state_dict = torch.load(os.path.join(args.savedir, "best_model.ckpt"))

    model = Model(
        in_dim=feat[0].size(0),
        hid_dim=args.hid_dim,
        out_dim=args.out_dim,
        num_nodes=len(feat),
        num_rels=2,
        dropout=0.,
        train=False)
    model.load_state_dict(state_dict)
    model.to(args.device).eval()
    with torch.no_grad():
        ptr_feat = model(g=graph, etypes=etypes, feat=feat).cpu().numpy()
    ptr_feat = {idx: ptr_feat[idx] for idx in range(len(ptr_feat))}
    with open(os.path.join(args.ptrdir, args.lm+".pkl"), "wb") as f:
        pickle.dump(ptr_feat, f)
    



    

