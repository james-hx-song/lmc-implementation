
class LRScheduler:
    def __init__(self, config):
        self.warmup_steps = config.warmup
        self.gamma = config.scheduler.gamma
        self.milestones = config.scheduler.milestones

        self.lr = config.lr
    
    def get_lr(self, iters):
        # Warmup
        if iters < self.warmup_steps:
            return self.lr * (iters + 1) / self.warmup_steps
        
        if iters in self.milestones:
            self.lr *= self.gamma

        return self.lr
        