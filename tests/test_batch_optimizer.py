from unittest.mock import patch
from greentensor.optimizers.batch_optimizer import BatchOptimizer, optimize_batch_size
from greentensor.utils.config import Config

def test_optimize_batch_cpu():
    # On CPU path: batch < min_batch_size → bumped to min
    config = Config(min_batch_size=32, max_batch_size=512)
    with patch("torch.cuda.is_available", return_value=False):
        result = optimize_batch_size(16, config)
    assert result == 32

def test_optimize_batch_cpu_doubles():
    config = Config(min_batch_size=32, max_batch_size=512)
    with patch("torch.cuda.is_available", return_value=False):
        result = optimize_batch_size(32, config)
    assert result == 64

def test_optimize_batch_respects_max():
    config = Config(min_batch_size=32, max_batch_size=64)
    with patch("torch.cuda.is_available", return_value=False):
        result = optimize_batch_size(64, config)
    assert result == 64

def test_batch_optimizer_revert():
    config = Config(min_batch_size=32, max_batch_size=512)
    with patch("torch.cuda.is_available", return_value=False):
        opt = BatchOptimizer(32, config)
        opt.apply()
        opt.revert()
    assert opt.optimal_batch == 32
