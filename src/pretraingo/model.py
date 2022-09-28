import torch
import torch.nn as nn
from dgl.nn import RelGraphConv


class RGCNEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_rels, dropout):
        super(RGCNEncoder, self).__init__()
        self.in_layer = RelGraphConv(in_dim, hid_dim, num_rels, activation=nn.ReLU(), dropout=dropout, layer_norm=True)
        self.hid_layer = RelGraphConv(hid_dim, hid_dim, num_rels, activation=nn.ReLU(), dropout=dropout, layer_norm=True)
        self.out_layer = RelGraphConv(hid_dim, out_dim, num_rels, dropout=dropout, layer_norm=True)

    def forward(self, g, etypes, feat):
        feat = self.in_layer(g, feat=feat, etypes=etypes)
        feat = self.hid_layer(g, feat=feat, etypes=etypes)
        feat = self.out_layer(g, feat=feat, etypes=etypes)        
        return feat


class Model(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_nodes, num_rels, dropout, train=True):
        super(Model, self).__init__()
        self.rgcn = RGCNEncoder(in_dim, hid_dim, out_dim, num_rels, dropout)

        self.do_train = train
        self.node_predicator = nn.Linear(out_dim, num_nodes)
        self.edge_predicator = nn.Linear(2*out_dim, num_rels+1)

    def forward(self, g, etypes, feat, node_mask=None, edge_mask=None):
        """
        Args:
            node_mask: 1-D LongTensor, node index for training
            edge_mask: 3-D LongTensor [src, tgt, label], indicates rels for training
        """
        feat = self.rgcn(g, etypes, feat)

        if self.do_train:
            node_logits = self.node_predicator(feat[node_mask])
            edge_logits = self.edge_predicator(torch.cat(
                (feat[edge_mask[:, 0]], feat[edge_mask[:, 1]]), dim=-1
            ))
            return node_logits, edge_logits
        else:
            return feat