"""Linear warmup + cosine annealing with warm restarts + gamma decay.

Self-contained re-implementation of the standalone
``linear_warmup_cosine_annealing_warm_restarts_weight_decay`` utility that is
normally shipped alongside the Lite-Mono training code. It is vendored here so
the RADIATE track is fully self-contained.

The scheduler is constructed by ``trainer.py`` as::

    ChainedScheduler(optimizer, T_0, T_mul=1, eta_min, last_epoch=-1,
                     max_lr, warmup_steps=0, gamma=1.0)
"""

import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


class ChainedScheduler(_LRScheduler):
    """Chains a linear warm-up phase with a cosine-annealing-with-warm-restarts
    schedule (optionally decaying the peak learning rate by ``gamma`` after each
    restart cycle)."""

    def __init__(self, optimizer, T_0, T_mul=1, eta_min=0, last_epoch=-1,
                 max_lr=1e-3, warmup_steps=0, gamma=1.0):
        if T_0 <= 0:
            raise ValueError("Expected T_0 > 0.")
        if T_mul < 1:
            raise ValueError("Expected T_mul >= 1.")
        self.T_0 = T_0
        self.T_mul = T_mul
        self.eta_min = eta_min
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            if self.warmup_steps == 0:
                return [self.max_lr for _ in self.base_lrs]
            alpha = self.last_epoch / self.warmup_steps
            return [self.eta_min + alpha * (self.max_lr - self.eta_min)
                    for _ in self.base_lrs]

        progress = self.last_epoch - self.warmup_steps
        T_cur = self.T_0
        cycle = 0
        while progress >= T_cur:
            progress -= T_cur
            cycle += 1
            T_cur = int(T_cur * self.T_mul)

        lr_scale = self.gamma ** cycle
        cosine = 0.5 * (1 + math.cos(math.pi * progress / T_cur))
        return [self.eta_min + lr_scale * (self.max_lr - self.eta_min) * cosine
                for _ in self.base_lrs]
