from mmengine.hooks import Hook
from mmseg.registry import HOOKS

@HOOKS.register_module()
class WarmupHook(Hook):
    def __init__(self, warmup_iters=5000):
        self.warmup_iters = warmup_iters

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        progress = min(runner.iter / self.warmup_iters, 1.0)
        for m in runner.model.modules():
            if hasattr(m, "warmup_progress"):
                m.warmup_progress = progress
