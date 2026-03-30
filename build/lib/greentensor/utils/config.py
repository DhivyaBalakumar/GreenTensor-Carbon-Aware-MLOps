from dataclasses import dataclass, field

@dataclass
class Config:
    # Batch optimization
    min_batch_size: int = 32
    max_batch_size: int = 512

    # GPU
    enable_cudnn_benchmark: bool = True
    enable_mixed_precision: bool = True

    # Idle detection
    idle_threshold_pct: float = 10.0   # GPU util % below which we consider it idle
    idle_sleep_s: float = 0.5          # seconds to sleep when idle detected

    # Reporting
    carbon_intensity_kg_per_kwh: float = 0.000233  # world avg kg CO2 per Wh → per kWh = 0.233 kg
