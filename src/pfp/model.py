import os
from turtle import forward
import torch
import pickle
import numpy as np
import torch.nn as nn

from utils import read_file


aa2idx = dict([
    ('A', 1), ('C', 2), ('D', 3), ('E', 4), ('F', 5), 
    ('G', 6), ('H', 7), ('I', 8), ('K', 9), ('L', 10),
    ('M', 11), ('N', 12), ('P', 13), ('Q', 14), ('R', 15),
    ('S', 16), ('T', 17), ('V', 18), ('W', 19), ('Y', 20)])


class SeqEmbedder(nn.Module):
    def __init__(self, args):
        super(SeqEmbedder, self).__init__()
        self.max_len = args.maxlen
        self.emb_size = args.seqembsize

        if args.seqemb is not None:     # pretrained embeddings
            self.seq_embs = {}
            self.seq_emb_dir = args.seqemb
        else:                           # without pretrained embeddings
            self.embedder = nn.Embedding(21, self.emb_size, padding_idx=0)  # 20+1
            self.seqs = dict(read_file(
                os.path.join(args.datadir, "seqs.tsv"), skip_header=False, split=True))

    def forward(self, pros, device):
        pro_lens = []
        if hasattr(self, "seq_embs"):
            embeddings = []
            for dbid in pros:
                if dbid not in self.seq_embs:
                    self.seq_embs[dbid] = np.load(os.path.join(self.seq_emb_dir, dbid+".npy"))
                emb = self.seq_embs[dbid]
                pl = emb.shape[0]
                pro_lens.append(pl)

                padded_emb = np.zeros((self.max_len, self.emb_size))
                padded_emb[:pl, :] = emb
                embeddings.append(padded_emb)
            return torch.Tensor(np.array(embeddings)).to(device), torch.LongTensor(pro_lens).to(device)
        else:
            tokens = []
            for dbid in pros:
                indexed_seq = [aa2idx[aa] for aa in self.seqs[dbid]]
                pl = len(indexed_seq)
                pro_lens.append(pl)

                tokens.append(indexed_seq+[0]*(self.max_len-pl))
            tokens = torch.LongTensor(tokens).to(device)
            return self.embedder(tokens), torch.LongTensor(pro_lens).to(device)


class StruEmbedder(nn.Module):
    def __init__(self, args):
        super(StruEmbedder, self).__init__()
        self.max_len = args.maxlen
        self.emb_size = args.struembsize
        self.stru_embs = {}
        self.stru_emb_dir = args.struemb
    
    def forward(self, pros, device):
        embeddings = []
        for dbid in pros:
            if dbid not in self.stru_embs:
                self.stru_embs[dbid] = torch.load(os.path.join(self.stru_emb_dir, dbid+"_mifst_per_tok.pt"))
            emb = self.stru_embs[dbid]

            padded_emb = np.zeros((self.max_len, self.emb_size))
            padded_emb[:emb.shape[0], :] = emb
            embeddings.append(padded_emb)

        return torch.Tensor(np.array(embeddings)).to(device)


class GoEmbedder(nn.Module):
    def __init__(self, args):
        super(GoEmbedder, self).__init__()
        self.emb_size = args.goembsize
        self.go_embs = pickle.load(open(args.goemb, "rb"))

    def forward(self, terms, device):
        embeddings = [self.go_embs[term] for term in terms]
        return torch.Tensor(np.array(embeddings)).to(device)
            

class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.hid_size = args.hidsize
    
    def forward(self, term_encoding, pro_encoding, pro_lens):
        """
        Args:
            pro_encoding    : pro_num x max_len x hid_size
            term_encoding   : term_num x hid_size
            pro_lens        : pro_num
        Output:
            out             : pro_num x term_num x hid_size
            attn            : pro_num x term_num x max_len
        """
        pro_num, max_len, _ = pro_encoding.size()
        term_num, _ = term_encoding.size()

        pro_encoding = pro_encoding.view(pro_num*max_len, -1)
        
        pro_encoding = pro_encoding.unsqueeze(1)    # pm x 1 x h
        term_encoding = term_encoding.unsqueeze(0)  # 1 x t x h

        attn = pro_encoding * term_encoding         # pm x t x h
        attn = torch.sum(attn, dim=-1)              # pm x t
        attn = attn.view(pro_num, max_len, term_num)    # p x m x t
        attn = attn.permute(0, 2, 1)                # p x t x m
        
        mask = torch.arange(max_len)\
            .repeat(pro_num, 1).to(pro_lens.device) # p x m
        mask = mask >= pro_lens.unsqueeze(-1)       # p x m
        mask = mask.unsqueeze(1).repeat(1, term_num, 1) # p x t x m
        
        attn = torch.masked_fill(attn, mask, float("-inf"))     # p x t x m
        attn = torch.softmax(attn, dim=-1)                      # p x t x m

        pro_encoding = pro_encoding.view(pro_num, max_len, -1)
        out = torch.matmul(attn, pro_encoding)                  # p x t x h
        return out, attn


class PFPModel(nn.Module):
    def __init__(self, args):
        super(PFPModel, self).__init__()
        self.seq_embedder = SeqEmbedder(args)
        if args.struemb is not None:
            self.stru_embedder = StruEmbedder(args)
        self.term_embedder = GoEmbedder(args)

        self.seq_mlp = nn.Sequential(nn.Linear(args.seqembsize, args.hidsize), nn.ReLU())
        self.stru_mlp = nn.Sequential(nn.Linear(args.struembsize, args.hidsize), nn.ReLU())

        self.pro_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=args.hidsize, nhead=4, \
                dim_feedforward=args.hidsize*4, dropout=args.dropout, batch_first=True), 
            num_layers=args.layers)
        self.term_enc = nn.Sequential(
            nn.Linear(args.goembsize, args.hidsize), 
            nn.Dropout(args.dropout), nn.ReLU(),
            nn.Linear(args.hidsize, args.hidsize), 
            nn.Dropout(args.dropout), nn.ReLU(),
            nn.Linear(args.hidsize, args.hidsize),
            nn.Dropout(args.dropout), nn.Tanh())
        
        self.attention = Attention(args)
        self.classifier = nn.Linear(args.hidsize, 1)

    def forward(self, ptm):
        device = next(self.parameters()).device
        pros, terms = ptm.index.values, ptm.columns.values

        seq_embs, pro_lens = self.seq_embedder(pros, device)
        seq_embs = self.seq_mlp(seq_embs)
        if hasattr(self, "stru_embedder"):
            stru_embs = self.stru_mlp(self.stru_embedder(pros, device))
            assert seq_embs.size() == stru_embs.size()
            pro_embs = seq_embs + stru_embs
        else:
            pro_embs = seq_embs

        pro_encoding = self.pro_enc(pro_embs)
        term_encoding = self.term_enc(self.term_embedder(terms, device))
        
        out, attn = self.attention(term_encoding, pro_encoding, pro_lens)
        logits = self.classifier(out).squeeze(-1)          # pro_num x term_num      

        return {
            "out": out,
            "logits": logits,
            "attn": attn}