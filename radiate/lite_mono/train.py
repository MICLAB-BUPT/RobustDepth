from __future__ import absolute_import, division, print_function

from options import LiteMonoOptions
from trainer import Trainer

import torch
import random
import numpy as np

def seed_all(seed):
    if not seed:
        seed = 1

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


options = LiteMonoOptions()
opts = options.parse()
seed_all(opts.pytorch_random_seed)

import pdb


if __name__ == "__main__":
    # pdb.set_trace()
    trainer = Trainer(opts)
    trainer.train()
