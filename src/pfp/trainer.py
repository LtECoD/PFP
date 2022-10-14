import os
import torch

from reader import PFPDataset
from model import PFPModel


class Trainer:
    def __init__(self, args):
        self.args = args

        self.trainset = PFPDataset(args, split="train")
        self.valset = PFPDataset(args, split="test")
        self.model = PFPModel(args)
        print(self.model)      

    def train(self):
        self.model.train()
        self.model.to(0)

        iterator = iter(self.trainset)
        for step in range(self.args.steps):
            data = next(iterator)
            output = self.model(data)

            raise NotImplementedError

    def valid(self):
        pass