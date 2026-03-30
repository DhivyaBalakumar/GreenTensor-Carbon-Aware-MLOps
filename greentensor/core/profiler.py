import subprocess
import platform
import time
import functools
from greentensor.utils.logger import logger

class Profiler:

    @staticmethod
    def get_gpu_metrics() -> dict:
        """
        Query live GPU utilization (%) and power draw (W) via nvidia-smi.
        Returns zeros gracefully when no GPU / nvidia-smi is unavailable.
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            line = result.stdout.strip()
            if line:
                parts = line.split(", ")
                util = float(parts[0])
                power = float(parts[1])
                return {"util_%": util, "power_W": power}
        except FileNotFoundError:
            pass  # nvidia-smi not installed
        except Exception as e:
            logger.debug(f"nvidia-smi query failed: {e}")
        return {"util_%": 0.0, "power_W": 0.0}

    @staticmethod
    def track_gpu(func):
        """Decorator that captures GPU metrics before and after a function call."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = Profiler.get_gpu_metrics()
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            end = Profiler.get_gpu_metrics()
            profile = {
                "duration_s": elapsed,
                "gpu_start": start,
                "gpu_end": end,
                "avg_power_W": (start["power_W"] + end["power_W"]) / 2,
            }
            return result, profile
        return wrapper

    @staticmethod
    def estimate_energy_kwh(power_w: float, duration_s: float) -> float:
        """Convert average power (W) and duration (s) to energy in kWh."""
        return (power_w * duration_s) / 3_600_000
