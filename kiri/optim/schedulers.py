"""kiri/optim/schedulers.py"""
import math


class _Scheduler:
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer   = optimizer
        self.last_epoch  = last_epoch
        self.base_lr     = optimizer.lr

    def step(self):
        self.last_epoch += 1
        self.optimizer.lr = self._get_lr()

    def get_lr(self): return self.optimizer.lr
    def _get_lr(self): raise NotImplementedError


class StepLR(_Scheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size; self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    def _get_lr(self):
        return self.base_lr * (self.gamma ** (self.last_epoch // self.step_size))


class MultiStepLR(_Scheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.milestones = sorted(milestones); self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    def _get_lr(self):
        n = sum(1 for m in self.milestones if self.last_epoch >= m)
        return self.base_lr * (self.gamma ** n)


class ExponentialLR(_Scheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma; super().__init__(optimizer, last_epoch)
    def _get_lr(self): return self.base_lr * (self.gamma ** self.last_epoch)


class CosineAnnealingLR(_Scheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max; self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    def _get_lr(self):
        t = self.last_epoch % self.T_max
        return self.eta_min + 0.5*(self.base_lr - self.eta_min)*(1 + math.cos(math.pi*t/self.T_max))


class LinearWarmup(_Scheduler):
    def __init__(self, optimizer, warmup_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs; super().__init__(optimizer, last_epoch)
    def _get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return self.base_lr * (self.last_epoch + 1) / self.warmup_epochs
        return self.base_lr


class ReduceLROnPlateau:
    def __init__(self, optimizer, mode="min", factor=0.5,
                 patience=5, min_lr=1e-7, verbose=True):
        self.optimizer = optimizer; self.mode = mode
        self.factor = factor; self.patience = patience
        self.min_lr = min_lr; self.verbose = verbose
        self._best = float("inf") if mode == "min" else float("-inf")
        self._wait = 0

    def step(self, metric):
        improved = (metric < self._best) if self.mode == "min" else (metric > self._best)
        if improved:
            self._best = metric; self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                old = self.optimizer.lr
                new = max(old * self.factor, self.min_lr)
                self.optimizer.lr = new
                if self.verbose and new != old:
                    print(f"  ReduceLR: {old:.2e} → {new:.2e}")
                self._wait = 0

    def get_lr(self): return self.optimizer.lr
