import os
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from itertools import chain
from itertools import combinations
from collections import defaultdict

from utils import read_file

#! 是否需要修改遍历方式，pro_first or term_first


class PFPDataset(Dataset):
    def __init__(self, args, split='train'):
        self.pos_samples = read_file(os.path.join(args.datadir, args.branch, split+".tsv"),
            skip_header=True, split=True)        
        self.pros = read_file(os.path.join(args.datadir, args.branch, split+"_pro.tsv"))
        self.terms = read_file(os.path.join(args.datadir, args.branch, split+"_term.tsv"))

        self.pro2term = defaultdict(list)
        self.term2pro = defaultdict(list)
        for (p, t) in self.pos_samples:
            self.pro2term[p].append(t)
            self.term2pro[t].append(p)  

        self.batch_mode = args.batchmode
        self.batch_pro = args.batchpro
        assert self.batch_pro % 2 == 0
        self.shuffle = args.shuffle

    def __iter__(self):
        if self.batch_mode == "term":
            term_combinations = list(combinations(self.terms, 2))
            if self.shuffle:
                random.shuffle(term_combinations)
            for terms in term_combinations:
                samples = list(chain(*[self._sample_term_proteins(t) for t in terms]))
                yield self.collater(samples)
        else:
            if self.shuffle:
                random.shuffle(self.pros)
            for idx in range(0, self.num_proteins, self.batch_pro):
                pros = self.pros[idx: min(self.num_proteins, idx+self.batch_pro)]        
                samples = list(chain(*[self._sample_protein_terms(p) for p in pros]))
                #! whether use all terms
                yield self.collater(samples, use_all_terms=False)
    
    def _sample_protein_terms(self, protein):
        """Get terms of a protein"""
        return [(protein, t) for t in self.pro2term[protein]]

    def _sample_term_proteins(self, term):
        """Sample proteins given a term"""
        pros = random.sample(self.term2pro[term], k=self.batch_pro//2)
        return [(p, term) for p in pros]
    
    def collater(self, samples, use_all_terms=False):
        """Build matrix of protein-term"""
        pros = set([p for p, t in samples])
        terms = set([t for p, t in samples]) if not use_all_terms else self.terms

        ptm = pd.DataFrame(columns=terms, index=pros, data=0)
        for p, t in samples:
            ptm.loc[p,t] = 1
        return ptm

    @property
    def num_terms(self):
        return len(self.terms)

    @property
    def num_proteins(self):
        return len(self.pros)