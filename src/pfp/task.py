import logging
from fairseq.tasks import register_task, LegacyFairseqTask

from src.pfp.utils import setup_seed
from src.pfp.reader import PFPDataset

logger = logging.getLogger(__name__)


@register_task("pfp")
class PFPTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        # data reader arguments
        parser.add_argument("--data-dir", type=str)
        parser.add_argument("--emb-dir", type=str)
        parser.add_argument("--max-len", type=int)

        parser.add_argument("--with-ca-coord", action="store_true", help="whether exploit ca coords of residues")
        parser.add_argument("--use-pretrain-emb", action="store_true", help="whether use pre-trained protein embedding")
    
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        setup_seed(args.seed)

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def load_dataset(self, split, combine=False, **kwargs):
        self.datasets[split] = PFPDataset(split, self.args)
    
    def reduce_metrics(self, logging_outputs, criterion):
        criterion.__class__.reduce_metrics(logging_outputs)
    
    def begin_epoch(self, epoch, model):
        for key in self.datasets:
            self.datasets[key].shuffle()