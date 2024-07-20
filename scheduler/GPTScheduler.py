from dataclasses import dataclass
import math

@dataclass
class LRConfig:
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 10
    max_steps: int = 50


class LRScheduler:

    def __init__(self, config):
        self.count = 0
        self.max_lr = config.max_lr
        self.min_lr = config.min_lr
        self.warmup_steps = config.warmup_steps
        self.max_steps = config.max_steps


    def get_lr(self,):
        # Linear Warmup
        if self.count < self.warmup_steps:
            return self.max_lr * (self.count + 1) / self.warmup_steps

        # min learning rate after decay is over
        if self.count > self.max_steps:
            return self.min_lr
    
        # Cosine Decay
        decay_ratio = (self.count - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

        self.count += 1
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

