import os
import dgl
import torch
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch import optim
import torch.nn.functional as F

from model import Model
from reader import read_graph


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, args):
        print(args)
        setup_seed(args.seed)
        self.save_dir = args.savedir
        self.result_dir = args.resultdir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        self.epoc = args.epoc
        self.device = args.device if hasattr(args, "device") else "cpu"
        self.data = read_graph(args)

        self.num_nodes = len(self.data[-1])
        self.rel_types = len(set(self.data[1]))
        self.num_edges = len(self.data[0])
        self.num_mask_nodes = int(self.num_nodes * args.nmask_ratio)
        self.num_mask_edges = int(self.num_edges * args.emask_ratio)

        print("The graph contains:"
              f"\n\t {self.num_nodes} nodes"
              f"\n\t {self.num_edges} edges"
              f"\n\t mask {self.num_mask_nodes} nodes"
              f"\n\t mask {self.num_mask_edges} edges")

        self.model = Model(
            in_dim=self.data[-1][0].size(0),
            hid_dim=args.hid_dim,
            out_dim=args.out_dim,
            num_nodes=self.num_nodes,
            num_rels=self.rel_types,
            dropout=args.dropout,
            train=True).to(self.device)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=args.lr)

        print(self.model)

    def train(self):
        losses = []
        nlosses = []
        elosses = []
        naccs = []
        eaccs = []

        min_loss = 1000
        epoc_iterator = tqdm(range(self.epoc))
        for epoc in epoc_iterator:
            (srcs,
            etypes,
            tgts,
            feat,
            node_mask,
            edge_mask,
            edge_label) = self.mask_graph()
            graph = dgl.graph((srcs, tgts), num_nodes=self.num_nodes, device=self.device)

            node_logits, edge_logits = self.model(
                g=graph,
                etypes=etypes,
                feat=feat,
                node_mask=node_mask,
                edge_mask=edge_mask)

            node_pred = torch.argmax(node_logits, dim=-1)
            edge_pred = torch.argmax(edge_logits, dim=-1)
            node_acc = (node_pred == node_mask).float().mean()
            edge_acc = (edge_pred == edge_label).float().mean()

            node_pred_loss = F.cross_entropy(node_logits, node_mask)
            edge_pred_loss = F.cross_entropy(edge_logits, edge_label)
            loss = node_pred_loss + edge_pred_loss

            epoc_iterator.set_description(
                f"loss:{round(float(loss), 3)} "
                f"NL:{round(float(node_pred_loss), 3)} "
                f"EL:{round(float(edge_pred_loss), 3)} "
                f"NACC:{round(float(node_acc), 4)} "
                f"EACC:{round(float(edge_acc), 4)}"
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(float(loss))
            nlosses.append(float(node_pred_loss))
            elosses.append(float(edge_pred_loss))
            naccs.append(float(node_acc))
            eaccs.append(float(edge_acc))
            
            if loss <= min_loss:
                min_loss = loss
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.ckpt"))
                with open(os.path.join(self.save_dir, "best_epoc.txt"), "w") as f:
                    f.write(
                        f"epoc: {epoc}\n"
                        f"loss: {float(loss)}\n"
                        f"nloss: {float(node_pred_loss)}\n"
                        f"nacc: {float(node_acc)}\n"
                        f"eacc: {float(edge_acc)}")
        
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, "last_model.ckpt"))
        with open(os.path.join(self.save_dir, "last_epoc.txt"), "w") as f:
            f.write(
                f"epoc: {epoc}\n"
                f"loss: {float(loss)}\n"
                f"nloss: {float(node_pred_loss)}\n"
                f"nacc: {float(node_acc)}\n"
                f"eacc: {float(edge_acc)}")

        with open(os.path.join(self.result_dir, "result.txt"), "w") as f:
            for idx in range(len(losses)):
                f.write(
                    f"loss: {round(losses[idx], 4)} "
                    f"nloss: {round(nlosses[idx], 4)} "
                    f"nacc: {round(naccs[idx])} "
                    f"eacc: {round(eaccs[idx])}\n")

    def mask_graph(self):
        srcs, etypes, tgts, feat = deepcopy(self.data)

        node_mask = sorted(random.sample(range(self.num_nodes), self.num_mask_nodes))
        # mask node
        for node in node_mask:
            # feat[node] = torch.zeros_like(feat[0])
            feat[node] = feat[node] + torch.randn_like(feat[0]) * 0.5

        edge_mask = []
        edge_label = []
        masked_edges = sorted(random.sample(range(len(srcs)), self.num_mask_edges))
        for idx in masked_edges:
            edge_mask.append([srcs[idx], tgts[idx]])
            edge_label.append(etypes[idx])
        # mask edge
        for idx, e in enumerate(masked_edges):
            del(srcs[e-idx])
            del(etypes[e-idx])
            del(tgts[e-idx])

        # sample negative edges
        tmp_graph = dgl.graph((srcs, tgts), num_nodes=self.num_nodes)
        neg_srcs, neg_tgts = dgl.sampling.global_uniform_negative_sampling(tmp_graph, num_samples=self.num_mask_edges)
        for idx, (s, t) in enumerate(zip(neg_srcs.numpy().tolist(), neg_tgts.numpy().tolist())):
            edge_mask.append([s, t])
            edge_label.append(self.rel_types)

        return (torch.LongTensor(srcs), 
            torch.LongTensor(etypes).to(self.device),
            torch.LongTensor(tgts), 
            torch.stack(feat, dim=0).to(self.device),
            torch.LongTensor(node_mask).to(self.device), 
            torch.LongTensor(edge_mask).to(self.device),
            torch.LongTensor(edge_label).to(self.device))