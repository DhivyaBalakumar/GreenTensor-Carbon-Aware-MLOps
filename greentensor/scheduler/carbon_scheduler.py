# -*- coding: utf-8 -*-
"""
Carbon-Aware Job Scheduler for GreenTensor.

Queries real-time grid carbon intensity via electricityMap API and
recommends the optimal time window to run a training job to minimize
carbon emissions — same compute, less CO2.

No API key required for the free public endpoint.
Falls back to static regional averages when API is unavailable.
"""

import time
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from greentensor.utils.logger import logger


# Static fallback carbon intensity (kg CO2/kWh) by region
# Source: electricityMap annual averages 2023
STATIC_INTENSITY = {
    "US-CAL-CISO":  0.000210,  # California
    "US-MIDA-PJM":  0.000320,  # Mid-Atlantic
    "US-TEX-ERCO":  0.000380,  # Texas
    "US-NW-PACW":   0.000120,  # Pacific Northwest (hydro)
    "GB":           0.000233,  # Great Britain
    "DE":           0.000350,  # Germany
    "FR":           0.000056,  # France (nuclear)
    "IN-NO":        0.000680,  # India North
    "IN-SO":        0.000620,  # India South
    "SG":           0.000408,  # Singapore
    "AU-NSW":       0.000490,  # Australia NSW
    "NO-NO1":       0.000017,  # Norway (hydro)
    "default":      0.000233,  # World average
}

# Renewable-heavy hours by region (UTC) — when grid is typically cleanest
CLEAN_HOURS_UTC = {
    "US-CAL-CISO":  [10, 11, 12, 13, 14],   # solar peak
    "US-TEX-ERCO":  [11, 12, 13, 14, 15],   # solar peak
    "GB":           [1, 2, 3, 4, 5],         # low demand, wind
    "DE":           [11, 12, 13, 14],        # solar peak
    "IN-NO":        [10, 11, 12, 13],        # solar peak
    "default":      [2, 3, 4, 5, 6],         # low demand globally
}


@dataclass
class GridSignal:
    """Real-time or estimated grid carbon intensity."""
    zone: str
    carbon_intensity_kg_per_kwh: float
    source: str                    # "electricitymap_api" | "static_fallback"
    timestamp: float
    is_low_carbon: bool            # below regional average
    renewable_pct: Optional[float] = None


@dataclass
class ScheduleRecommendation:
    """Recommendation for when to run a training job."""
    run_now: bool
    current_intensity: float
    recommended_delay_hours: float
    estimated_intensity_at_recommended_time: float
    carbon_savings_pct: float
    reason: str
    optimal_window_utc: Optional[str] = None
    estimated_savings_kg: Optional[float] = None  # for a given job size


class CarbonAwareScheduler:
    """
    Queries real-time grid carbon intensity and recommends
    the optimal time to run a training job.

    Usage:
        scheduler = CarbonAwareScheduler(zone="US-CAL-CISO")

        # Check before submitting a job
        rec = scheduler.should_run_now(estimated_duration_hours=4)
        if rec.run_now:
            train()
        else:
            print(rec.reason)
            # Schedule for rec.recommended_delay_hours from now

        # Or use as a context manager that waits for clean grid
        with scheduler.wait_for_clean_grid(max_wait_hours=6):
            train()
    """

    ELECTRICITYMAP_URL = "https://api.electricitymap.org/v3/carbon-intensity/latest?zone={zone}"

    def __init__(
        self,
        zone: str = "default",
        low_carbon_threshold_pct: float = 20.0,  # % below regional average = "low carbon"
        api_token: Optional[str] = None,
    ):
        self.zone = zone
        self.low_carbon_threshold_pct = low_carbon_threshold_pct
        self.api_token = api_token
        self._cache: Optional[GridSignal] = None
        self._cache_ttl = 300  # 5 minutes

    def get_current_intensity(self) -> GridSignal:
        """Get current grid carbon intensity. Uses API if available, falls back to static."""
        if self._cache and (time.time() - self._cache.timestamp) < self._cache_ttl:
            return self._cache

        signal = self._fetch_from_api() or self._static_signal()
        self._cache = signal
        return signal

    def should_run_now(
        self,
        estimated_duration_hours: float = 1.0,
        estimated_energy_kwh: Optional[float] = None,
        max_acceptable_intensity: Optional[float] = None,
    ) -> ScheduleRecommendation:
        """
        Decide whether to run a job now or wait for a cleaner grid.

        Parameters
        ----------
        estimated_duration_hours  : how long the job will run
        estimated_energy_kwh      : estimated energy consumption (for savings calc)
        max_acceptable_intensity  : override threshold (kg CO2/kWh)
        """
        signal = self.get_current_intensity()
        regional_avg = STATIC_INTENSITY.get(self.zone, STATIC_INTENSITY["default"])
        threshold = max_acceptable_intensity or (regional_avg * (1 - self.low_carbon_threshold_pct / 100))

        clean_hours = CLEAN_HOURS_UTC.get(self.zone, CLEAN_HOURS_UTC["default"])
        current_hour_utc = time.gmtime().tm_hour
        hours_to_clean = self._hours_until_clean_window(current_hour_utc, clean_hours)

        est_clean_intensity = regional_avg * 0.65  # clean window is ~35% below average

        if signal.carbon_intensity_kg_per_kwh <= threshold:
            savings_kg = None
            if estimated_energy_kwh:
                savings_kg = 0.0  # already running at optimal time
            return ScheduleRecommendation(
                run_now=True,
                current_intensity=signal.carbon_intensity_kg_per_kwh,
                recommended_delay_hours=0.0,
                estimated_intensity_at_recommended_time=signal.carbon_intensity_kg_per_kwh,
                carbon_savings_pct=0.0,
                reason=(
                    f"Grid is clean now ({signal.carbon_intensity_kg_per_kwh*1000:.1f} gCO2/kWh). "
                    f"Run immediately. Source: {signal.source}."
                ),
                estimated_savings_kg=savings_kg,
            )
        else:
            savings_pct = ((signal.carbon_intensity_kg_per_kwh - est_clean_intensity)
                          / signal.carbon_intensity_kg_per_kwh * 100)
            savings_kg = None
            if estimated_energy_kwh:
                savings_kg = estimated_energy_kwh * (
                    signal.carbon_intensity_kg_per_kwh - est_clean_intensity
                )
            clean_window_str = f"{clean_hours[0]:02d}:00-{clean_hours[-1]+1:02d}:00 UTC"
            return ScheduleRecommendation(
                run_now=False,
                current_intensity=signal.carbon_intensity_kg_per_kwh,
                recommended_delay_hours=hours_to_clean,
                estimated_intensity_at_recommended_time=est_clean_intensity,
                carbon_savings_pct=savings_pct,
                reason=(
                    f"Grid is carbon-heavy now ({signal.carbon_intensity_kg_per_kwh*1000:.1f} gCO2/kWh). "
                    f"Wait {hours_to_clean:.1f}h for clean window ({clean_window_str}). "
                    f"Estimated {savings_pct:.1f}% carbon reduction."
                ),
                optimal_window_utc=clean_window_str,
                estimated_savings_kg=savings_kg,
            )

    def wait_for_clean_grid(
        self,
        max_wait_hours: float = 6.0,
        check_interval_minutes: float = 15.0,
    ):
        """
        Context manager that waits until the grid is clean before running.
        Checks every check_interval_minutes, up to max_wait_hours.

        Usage:
            with scheduler.wait_for_clean_grid(max_wait_hours=4):
                train()
        """
        return _CleanGridContext(self, max_wait_hours, check_interval_minutes)

    def print_recommendation(self, estimated_duration_hours: float = 1.0,
                             estimated_energy_kwh: Optional[float] = None):
        """Print a human-readable scheduling recommendation."""
        rec = self.should_run_now(estimated_duration_hours, estimated_energy_kwh)
        print("\n  +-- Carbon-Aware Scheduler ----------------+")
        print(f"  Zone             : {self.zone}")
        print(f"  Current intensity: {rec.current_intensity*1000:.1f} gCO2/kWh")
        print(f"  Recommendation   : {'RUN NOW' if rec.run_now else 'WAIT'}")
        if not rec.run_now:
            print(f"  Delay            : {rec.recommended_delay_hours:.1f} hours")
            print(f"  Optimal window   : {rec.optimal_window_utc}")
            print(f"  Carbon savings   : {rec.carbon_savings_pct:.1f}%")
            if rec.estimated_savings_kg:
                print(f"  CO2 avoided      : {rec.estimated_savings_kg:.6f} kg")
        print(f"  Reason           : {rec.reason}")
        print("  +------------------------------------------+\n")
        return rec

    # ── Internal ───────────────────────────────────────────────────────────────

    def _fetch_from_api(self) -> Optional[GridSignal]:
        if self.zone == "default":
            return None
        try:
            url = self.ELECTRICITYMAP_URL.format(zone=self.zone)
            req = urllib.request.Request(url)
            if self.api_token:
                req.add_header("auth-token", self.api_token)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            intensity = data.get("carbonIntensity", 0) / 1000  # gCO2/kWh -> kg
            regional_avg = STATIC_INTENSITY.get(self.zone, STATIC_INTENSITY["default"])
            return GridSignal(
                zone=self.zone,
                carbon_intensity_kg_per_kwh=intensity,
                source="electricitymap_api",
                timestamp=time.time(),
                is_low_carbon=intensity < regional_avg * 0.8,
                renewable_pct=data.get("renewablePercentage"),
            )
        except Exception as e:
            logger.debug(f"electricityMap API unavailable: {e} — using static fallback")
            return None

    def _static_signal(self) -> GridSignal:
        intensity = STATIC_INTENSITY.get(self.zone, STATIC_INTENSITY["default"])
        regional_avg = STATIC_INTENSITY.get(self.zone, STATIC_INTENSITY["default"])
        # Simulate time-of-day variation (±20%)
        hour = time.gmtime().tm_hour
        clean_hours = CLEAN_HOURS_UTC.get(self.zone, CLEAN_HOURS_UTC["default"])
        if hour in clean_hours:
            intensity *= 0.75  # cleaner during renewable peak
        elif hour in range(17, 21):
            intensity *= 1.20  # evening demand peak
        return GridSignal(
            zone=self.zone,
            carbon_intensity_kg_per_kwh=intensity,
            source="static_fallback",
            timestamp=time.time(),
            is_low_carbon=intensity < regional_avg * 0.8,
        )

    @staticmethod
    def _hours_until_clean_window(current_hour: int, clean_hours: List[int]) -> float:
        for h in sorted(clean_hours):
            if h > current_hour:
                return float(h - current_hour)
        # Next day
        return float(24 - current_hour + sorted(clean_hours)[0])


class _CleanGridContext:
    def __init__(self, scheduler: CarbonAwareScheduler,
                 max_wait_hours: float, check_interval_minutes: float):
        self.scheduler = scheduler
        self.max_wait_hours = max_wait_hours
        self.check_interval_s = check_interval_minutes * 60
        self._waited = 0.0

    def __enter__(self):
        deadline = time.time() + self.max_wait_hours * 3600
        while time.time() < deadline:
            rec = self.scheduler.should_run_now()
            if rec.run_now:
                logger.info(
                    f"Grid is clean ({rec.current_intensity*1000:.1f} gCO2/kWh). Starting job."
                )
                return self
            logger.info(
                f"Grid is carbon-heavy ({rec.current_intensity*1000:.1f} gCO2/kWh). "
                f"Waiting {self.check_interval_s/60:.0f} min. {rec.reason}"
            )
            time.sleep(self.check_interval_s)
            self._waited += self.check_interval_s / 3600
        logger.warning(
            f"Max wait time ({self.max_wait_hours}h) reached. Running job regardless of grid state."
        )
        return self

    def __exit__(self, *args):
        pass
