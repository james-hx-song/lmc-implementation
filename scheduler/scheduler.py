
class LRScheduler:
    def __init__(self, config):
        self.warmup_steps = config.warmup
        self.gamma = config.scheduler.gamma
        self.milestones = config.scheduler.milestones

        self.lr = config.lr
        self.steps = 0
    
    def get_lr(self):
        # Warmup
        if self.steps < self.warmup_steps:
            return self.lr * (self.steps + 1) / self.warmup_steps
        
        if self.steps in self.milestones:
            self.lr *= self.gamma

        return self.lr
        