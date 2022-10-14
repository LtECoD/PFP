import torch
import random
import logging
import argparse
import numpy as np

from trainer import Trainer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # reader arguments
    parser.add_argument("--branch", type=str, help="category of go graph")
    parser.add_argument("--datadir", type=str)
    parser.add_argument("--maxlen", type=int, help="max length of protein sequence")
    parser.add_argument("--batchmode", type=str, choices=["term", "protein"],
        help="gather batch data in term-centric or protein-centric")
    parser.add_argument("--batchpro", type=int, help="Protein nums in a batch")
    parser.add_argument("--shuffle", action="store_true")

    # model arguments
    parser.add_argument('--seqemb', type=str, help="directory storing protein embeddings")
    parser.add_argument("--struemb", type=str, help="directory storing protein structure embeddings")
    parser.add_argument("--goemb", type=str, help="directory storing GO term embeddings")
    parser.add_argument("--seqembsize", type=int)
    parser.add_argument("--struembsize", type=int)
    parser.add_argument("--goembsize", type=int)
    parser.add_argument("--hidsize", type=int)
    parser.add_argument("--layers", type=int, help="layer nums of protein encoder, layer of term encoder is 3")
    parser.add_argument("--dropout", type=float)

    # trainer arguments
    parser.add_argument("--device", type=int, nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--savedir", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--validinterval", type=int, help="step intervals to save model")

    args = parser.parse_args()
    setup_seed(args.seed)
    print(args)

    trainer = Trainer(args)
    trainer.train()