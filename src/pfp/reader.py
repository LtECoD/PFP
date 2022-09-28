import os
import re
import torch
import random
import numpy as np
from fairseq.data import FairseqDataset


def read_coords(line):
    dbid, coord_str = line.strip().split("\t")
    coord = np.array(list(map(float, coord_str.strip().split()))).reshape(-1, 3)
    return dbid, coord


def read_gos(line):
    try:
        dbid, go_str = line.strip().split("\t")
        gos = go_str.split()
    except ValueError:
        gos = []
    return dbid, gos


class PFPDataset(FairseqDataset):
    def __init__(self, split, args):
        self.split = split

        # read protein sequences
        with open(os.path.join(args.data_dir, split+"_seq.tsv"), "r") as f:
            self.seqs = dict([line.strip().split("\t") for line in f.readlines()])
        # read molecular function GO terms
        with open(os.path.join(args.data_dir, split+"_mf.tsv"), "r") as f:
            self.mfs = dict([read_gos(line) for line in f.readlines()])
        # read cc GO terms
        with open(os.path.join(args.data_dir, split+"_cc.tsv"), "r") as f:
            self.ccs = dict([read_gos(line) for line in f.readlines()])
        # read bp Go terms
        with open(os.path.join(args.data_dir, split+"_bp.tsv"), "r") as f:
            self.bps = dict([read_gos(line) for line in f.readlines()])
        if args.with_ca_coord:
            with open(os.path.join(args.data_dir, split+"_coord.tsv"), "r") as f:
                self.coords = dict([read_coords(line) for line in f.readlines()])
        self.dbids = list(self.seqs.keys())

        self.max_len = args.max_len
        if args.use_pretrained_emb:
            self.emb_dim = args.emb_dim
            self.embs = {}

        self.print_statistics()

    def get_embed(self, pro):
        emb = np.load(os.path.join(self.emb_dir, pro+".npy"))
        pro_len = emb.shape[0]
        padded_emb = np.zeros((self.max_len, self.emb_dim))
        padded_emb[:pro_len, :] = emb
        return padded_emb, pro_len

    def print_statistics(self):
        num_of_proteins = len(self.dbids)
        avg_len = sum([len(seq) for seq in self.seqs.values()]) / num_of_proteins
        avg_mfs = sum([len(gos) for gos in self.mf.values()]) / num_of_proteins
        avg_bps = sum([len(gos) for gos in self.bp.values()]) / num_of_proteins
        avg_ccs = sum([len(gos) for gos in self.cc.values()]) / num_of_proteins

        print(f"========{self.split} Dataset Statistics========")
        print(f"Number of Proteins: {num_of_proteins}")
        print(f"Avg Squence Length: {round(avg_len, 3)}")
        print(f"Avg MF Terms: {round(avg_mfs, 3)}")
        print(f"Avg BP Terms: {round(avg_bps, 3)}")
        print(f"Avg CC Terms: {round(avg_ccs, 3)}")

#     def __getitem__(self, index):
#         fpro, spro, label = self.samples[index]
#         fpro_seq = self.acid_seqs[fpro]
#         spro_seq = self.acid_seqs[spro]

#         fpro_emb, fpro_len = self.get_embed(fpro)
#         spro_emb, spro_len = self.get_embed(spro)
#         assert fpro_len == len(fpro_seq)
#         assert spro_len == len(spro_seq)

#         sample_dict = {
#             "index": index,
#             "fpro": fpro,
#             "spro": spro,
#             "femb": fpro_emb,
#             "semb": spro_emb,
#             'fprolen': fpro_len,
#             'sprolen': spro_len,
#             "fseq": fpro_seq,
#             "sseq": spro_seq,
#             "label": label}
#         return sample_dict
    
#     def __len__(self):
#         return len(self.samples)
    
#     def num_tokens(self, index):
#         """Return the number of tokens in a sample. This value is used to
#         enforce ``--max-tokens`` during batching."""
#         fpro, spro, _ = self.samples[index]
#         return len(self.acid_seqs[fpro]) + len(self.acid_seqs[spro])

#     def collater(self, samples):
#         labels = torch.LongTensor([int(sample['label']) for sample in samples])
#         fst_embs = np.array([sample["femb"] for sample in samples])
#         sec_embs = np.array([sample["semb"] for sample in samples])
#         fst_lens = [sample["fprolen"] for sample in samples]
#         sec_lens = [sample["sprolen"] for sample in samples]

#         model_inputs = {
#             "fst_embs": torch.Tensor(fst_embs),
#             "fst_lens": torch.LongTensor(fst_lens),
#             "sec_embs": torch.Tensor(sec_embs),
#             "sec_lens": torch.LongTensor(sec_lens)}

#         fpros = [sample["fpro"] for sample in samples]
#         spros = [sample["spro"] for sample in samples]
#         fseqs = [sample["fseq"] for sample in samples]
#         sseqs = [sample["sseq"] for sample in samples]
#         data_info = {"fpros": fpros, \
#             "spros": spros, "fseqs": fseqs, "sseqs": sseqs}
#         return {
#             "inputs": model_inputs,
#             "labels": labels,
#             "infos": data_info}
    
#     def shuffle(self):
#         random.shuffle(self.samples)


# standard_acids = [
#         ('A', 1), ('C', 6), ('D', 5), ('E', 7), ('F', 2), 
#         ('G', 1), ('H', 4), ('I', 2), ('K', 5), ('L', 2),
#         ('M', 3), ('N', 4), ('P', 2), ('Q', 7), ('R', 4),
#         ('S', 3), ('T', 3), ('V', 1), ('W', 4), ('Y', 3), ('X', 0)]


# class OriPPIDataset(PPIDataset):
#     def __init__(self, split, args):
#         super().__init__(split, args)
#         self.vocab = {k[0]: idx+1 for idx, k in enumerate(standard_acids)}
#         self.pad_idx = 0

#         self.indexed_seqs = {}
#         for key, seq in self.acid_seqs.items():
#             seq = re.sub(r"[UZOB]", "X", seq)
#             idxseq = [self.vocab[a] for a in list(seq)]
#             idxseq = idxseq + [self.pad_idx] * (args.max_len-len(idxseq))
#             self.indexed_seqs[key] = idxseq

#     def __getitem__(self, index):
#         fpro, spro, label = self.samples[index]
#         fpro_seq = self.acid_seqs[fpro]
#         spro_seq = self.acid_seqs[spro]

#         fpro_indexed_seq = self.indexed_seqs[fpro]
#         spro_indexed_seq = self.indexed_seqs[spro]
        
#         fpro_len = len(fpro_seq)
#         spro_len = len(spro_seq)

#         sample_dict = {
#             "index": index,
#             "fpro": fpro,
#             "spro": spro,
#             "fproindexedseq": fpro_indexed_seq,
#             "sproindexedseq": spro_indexed_seq,
#             'fprolen': fpro_len,
#             'sprolen': spro_len,
#             "fseq": fpro_seq,
#             "sseq": spro_seq,
#             "label": label}
#         return sample_dict


#     def collater(self, samples):
#         labels = torch.LongTensor([int(sample['label']) for sample in samples])
#         fst_seqs = np.array([sample["fproindexedseq"] for sample in samples])
#         sec_seqs = np.array([sample["sproindexedseq"] for sample in samples])

#         fst_lens = [sample["fprolen"] for sample in samples]
#         sec_lens = [sample["sprolen"] for sample in samples]

#         model_inputs = {
#             "fst_seqs": torch.LongTensor(fst_seqs),
#             "fst_lens": torch.LongTensor(fst_lens),
#             "sec_seqs": torch.LongTensor(sec_seqs),
#             "sec_lens": torch.LongTensor(sec_lens)}

#         fpros = [sample["fpro"] for sample in samples]
#         spros = [sample["spro"] for sample in samples]
#         fseqs = [sample["fseq"] for sample in samples]
#         sseqs = [sample["sseq"] for sample in samples]
#         data_info = {"fpros": fpros, \
#             "spros": spros, "fseqs": fseqs, "sseqs": sseqs}
#         return {
#             "inputs": model_inputs,
#             "labels": labels,
#             "infos": data_info}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data/processed/quickgo")
    parser.add_argument("--emb-dir", type=str, default="./data/processed/embed")
    parser.add_argument("--max-len", type=int, default=800)

    parser.add_argument("--with-ca-coord", action="store_true", help="whether exploit ca coords of residues")
    parser.add_argument("--use-pretrain-emb", action="store_true", help="whether use pre-trained protein embedding")
    args = parser.parse_args()

    dataset = PFPDataset('test', args)

