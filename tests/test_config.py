from greentensor.utils.config import Config

def test_defaults():
    c = Config()
    assert c.min_batch_size == 32
    assert c.max_batch_size == 512
    assert c.enable_cudnn_benchmark is True
    assert c.enable_mixed_precision is True
    assert c.idle_threshold_pct == 10.0
    assert c.carbon_intensity_kg_per_kwh == 0.000233

def test_custom():
    c = Config(min_batch_size=16, max_batch_size=256)
    assert c.min_batch_size == 16
    assert c.max_batch_size == 256
