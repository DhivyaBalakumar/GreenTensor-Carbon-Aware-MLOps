"""
GreenTensor Batch Optimizer — GPU memory-aware batch size recommendation.

Author: Dhivya Balakumar <dhivyabalakumar28@gmail.com>
License: MIT
"""
from .base import BaseOptimizer
from greentensor.utils.config import Config
from greentensor.utils.logger import logger


class BatchOptimizer(BaseOptimizer):
    """Suggests an optimal batch size based on GPU memory availability."""

    def __init__(self, current_batch: int, config: Config = None):
        self.config = config or Config()
        self.original_batch = current_batch
        self.optimal_batch = current_batch

    def apply(self):
        self.optimal_batch = self._compute_optimal(self.original_batch)
        if self.optimal_batch != self.original_batch:
            logger.info(f"Batch size optimized: {self.original_batch} → {self.optimal_batch}")
        return self.optimal_batch

    def revert(self):
        self.optimal_batch = self.original_batch

    def _compute_optimal(self, batch: int) -> int:
        import torch
        if torch.cuda.is_available():
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_gb = free_mem / (1024 ** 3)
            # scale batch size proportionally to free memory, capped by config
            if free_gb > 4:
                candidate = min(batch * 4, self.config.max_batch_size)
            elif free_gb > 2:
                candidate = min(batch * 2, self.config.max_batch_size)
            else:
                candidate = max(batch, self.config.min_batch_size)
            return candidate
        # CPU: just double if below min threshold
        if batch < self.config.min_batch_size:
            return self.config.min_batch_size
        if batch < 64:
            return min(batch * 2, self.config.max_batch_size)
        return batch


def optimize_batch_size(current_batch: int, config: Config = None) -> int:
    """Convenience function — returns the recommended batch size."""
    return BatchOptimizer(current_batch, config).apply()
