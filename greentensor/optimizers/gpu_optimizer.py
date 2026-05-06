"""
GreenTensor GPU Optimizer — cuDNN benchmark and mixed precision.

Author: Dhivya Balakumar <dhivyabalakumar28@gmail.com>
License: MIT
"""
import torch
from greentensor.utils.logger import logger
from greentensor.utils.config import Config

class GPUOptimizer(BaseOptimizer):
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self._prev_cudnn_benchmark = None
        self._prev_cudnn_enabled = None

    def apply(self):
        if not torch.cuda.is_available():
            logger.warning("No CUDA GPU detected — GPU optimizations skipped.")
            return

        if self.config.enable_cudnn_benchmark:
            self._prev_cudnn_benchmark = torch.backends.cudnn.benchmark
            self._prev_cudnn_enabled = torch.backends.cudnn.enabled
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            logger.info("cuDNN benchmark mode enabled.")

    def revert(self):
        if not torch.cuda.is_available():
            return
        if self._prev_cudnn_benchmark is not None:
            torch.backends.cudnn.benchmark = self._prev_cudnn_benchmark
            torch.backends.cudnn.enabled = self._prev_cudnn_enabled
            logger.info("cuDNN settings restored.")

    def mixed_precision(self):
        """Returns an autocast context for mixed precision training."""
        if torch.cuda.is_available() and self.config.enable_mixed_precision:
            return torch.cuda.amp.autocast()
        # no-op context on CPU
        import contextlib
        return contextlib.nullcontext()
