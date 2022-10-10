import os
import torch
import pickle
import logging
import numpy as np
from fairseq.tasks import register_task, LegacyFairseqTask

from pfp.utils import setup_seed
from pfp.reader import PFPDataset

logger = logging.getLogger(__name__)


class EmbedderHub:
    def __init__(self, seq_emb_dir, stru_emb_dir=None, go_emb_file=None):
        self.seq_emb_dir = seq_emb_dir
        self.stru_emb_dir = stru_emb_dir

        self.seq_embs = {}
        if stru_emb_dir is not None:
            self.stru_embs = {}
        if go_emb_file is not None:
            gemb = pickle.load(open(go_emb_file, "rb"))
            self.go_embs = {k: torch.Tensor(v) for k, v in gemb.items()}

    def get_pro_emb(self, dbid):
        if dbid not in self.seq_embs:
            emb = np.load(os.path.join(self.seq_emb_dir, dbid+".npy"))
            self.seq_embs[dbid] = torch.Tensor(emb)
        seq_emb = self.seq_embs[dbid]

        if hasattr(self, "stru_embs"):
            if dbid not in self.stru_emb_dir:
                emb = torch.load(os.path.join(self.stru_emb_dir, dbid+"_mifst_per_tok.pt"))
                self.stru_embs[dbid] = emb
            stru_emb = self.stru_embs[dbid]
        else:
            stru_emb = None
        return seq_emb, stru_emb


@register_task("pfp")
class PFPTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        # data reader arguments
        parser.add_argument("--branch", type=str, help="category of go graph")
        parser.add_argument("--datadir", type=str)
        parser.add_argument('--seqembdir', type=str, help="directory storing protein embeddings")
        parser.add_argument("--struembdir", type=str, help="directory storing protein structure embeddings")
        parser.add_argument("--goembfile", type=str, help="directory storing GO term embeddings")
        parser.add_argument("--maxlen", type=int, help="max length of protein sequence")

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.emb_hub = EmbedderHub(args.seqembdir, args.struembdir, args.goembfile)
        setup_seed(args.seed)

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def load_dataset(self, split, combine=False, **kwargs):
        self.datasets[split] = PFPDataset(split, self.args, emb_hub=self.emb_hub)

    def reduce_metrics(self, logging_outputs, criterion):
        criterion.__class__.reduce_metrics(logging_outputs)