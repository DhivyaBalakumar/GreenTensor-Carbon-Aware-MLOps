import time
from greentensor.utils.config import Config
from greentensor.utils.logger import logger

class Tracker:
    """
    Tracks energy and carbon emissions for a workload.
    Uses CodeCarbon when available; falls back to nvidia-smi power sampling.
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self._codecarbon = None
        self._start_time = None
        self._power_samples: list[float] = []
        self._sampling = False
        self._sample_thread = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self):
        self._start_time = time.perf_counter()
        if self._try_start_codecarbon():
            logger.info("Tracking emissions via CodeCarbon.")
        else:
            logger.info("CodeCarbon unavailable — falling back to nvidia-smi power sampling.")
            self._start_power_sampling()

    def stop(self) -> tuple[float, float]:
        """
        Returns (emissions_kg_co2, energy_kwh).
        """
        duration = time.perf_counter() - self._start_time

        if self._codecarbon:
            return self._stop_codecarbon()

        return self._stop_power_sampling(duration)

    # ------------------------------------------------------------------ #
    #  CodeCarbon path                                                     #
    # ------------------------------------------------------------------ #

    def _try_start_codecarbon(self) -> bool:
        try:
            from codecarbon import EmissionsTracker
            self._codecarbon = EmissionsTracker(
                log_level="error",
                save_to_file=False,
                save_to_api=False,
            )
            self._codecarbon.start()
            return True
        except Exception as e:
            logger.debug(f"CodeCarbon init failed: {e}")
            self._codecarbon = None
            return False

    def _stop_codecarbon(self) -> tuple[float, float]:
        emissions_kg = self._codecarbon.stop() or 0.0
        # CodeCarbon exposes energy via its internal tracker
        try:
            energy_kwh = self._codecarbon._total_energy.kWh
        except Exception:
            # fallback: derive from emissions using config intensity
            energy_kwh = emissions_kg / self.config.carbon_intensity_kg_per_kwh
        return emissions_kg, energy_kwh

    # ------------------------------------------------------------------ #
    #  Power-sampling fallback                                             #
    # ------------------------------------------------------------------ #

    def _start_power_sampling(self):
        import threading
        self._sampling = True
        self._power_samples = []

        def _sample():
            from greentensor.core.profiler import Profiler
            while self._sampling:
                m = Profiler.get_gpu_metrics()
                self._power_samples.append(m["power_W"])
                time.sleep(0.5)

        self._sample_thread = threading.Thread(target=_sample, daemon=True)
        self._sample_thread.start()

    def _stop_power_sampling(self, duration_s: float) -> tuple[float, float]:
        self._sampling = False
        if self._sample_thread:
            self._sample_thread.join(timeout=2)

        if self._power_samples:
            avg_power_w = sum(self._power_samples) / len(self._power_samples)
        else:
            avg_power_w = 0.0

        energy_kwh = (avg_power_w * duration_s) / 3_600_000
        emissions_kg = energy_kwh * self.config.carbon_intensity_kg_per_kwh
        return emissions_kg, energy_kwh
