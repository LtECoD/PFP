import argparse

from trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", type=str, help="language model")
    parser.add_argument("--godir", type=str, help="input go graph directory")
    parser.add_argument("--embdir", type=str, help="dir to store initial embedding")
    parser.add_argument("--savedir", type=str)
    parser.add_argument("--resultdir", type=str)

    # model parameters
    parser.add_argument("--hid_dim", type=int, help="dimension of hidden")
    parser.add_argument("--out_dim", type=int, help="output dimension")
    parser.add_argument("--dropout", type=float)
 
    # train parameters
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epoc", type=int)
    parser.add_argument("--device", type=int)
    parser.add_argument("--nmask_ratio", type=float)
    parser.add_argument("--emask_ratio", type=float)

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()



