from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS

import os
import numpy as np

@METRICS.register_module()
class ConfusionMatrixMetric(BaseMetric):

    default_prefix = 'confusion'

    def __init__(self,
                 num_classes,
                 ignore_index=255,
                 collect_device='cpu',
                 prefix=None):

        super().__init__(
            collect_device=collect_device,
            prefix=prefix)

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        if os.path.exists("confusion_matrix.npy"):
            self.cm = np.load("confusion_matrix.npy")
        else:
            self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            pred = sample['pred_sem_seg']['data'].squeeze().flatten().cpu().numpy().astype(np.uint8)
            gt = sample['gt_sem_seg']['data'].squeeze().flatten().cpu().numpy().astype(np.uint8)
            mask = gt != self.ignore_index

            pred = pred[mask]
            gt = gt[mask]

            inds = self.num_classes * gt + pred
            mat = np.bincount(
                inds,
                minlength=self.num_classes**2
            ).reshape(self.num_classes, self.num_classes)

            self.cm += mat

    def compute_metrics(self, results):
        np.save("confusion_matrix.npy", self.cm)
        return {}

