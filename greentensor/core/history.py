# -*- coding: utf-8 -*-
"""
Run history — persists RunMetrics across multiple sessions to a JSON file.
Used by the ESG reporter and the dashboard history view.
"""
import json
import time
import os
from dataclasses import asdict
from typing import List, Optional
from greentensor.report.metrics import RunMetrics
from greentensor.utils.logger import logger

DEFAULT_HISTORY_PATH = "greentensor_history.json"


class RunHistory:
    """
    Persists RunMetrics to disk across sessions.
    Each entry includes the metrics plus metadata (model name, stage, timestamp).
    """

    def __init__(self, path: str = DEFAULT_HISTORY_PATH):
        self.path = path
        self._records: List[dict] = []
        self._load()

    def record(self, metrics: RunMetrics, model_name: str = "unknown",
               stage: str = "training", tags: Optional[dict] = None):
        entry = {
            "timestamp": time.time(),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "stage": stage,
            "tags": tags or {},
            "duration_s": metrics.duration_s,
            "energy_kwh": metrics.energy_kwh,
            "emissions_kg": metrics.emissions_kg,
            "idle_seconds": metrics.idle_seconds,
        }
        self._records.append(entry)
        self._save()
        logger.info(f"Run recorded to history: {model_name} | {metrics.energy_kwh:.6f} kWh")
        return entry

    def all(self) -> List[dict]:
        return list(self._records)

    def last(self, n: int = 10) -> List[dict]:
        return self._records[-n:]

    def clear(self):
        self._records = []
        self._save()

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._records, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save run history: {e}")

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._records = json.load(f)
            except Exception:
                self._records = []
        else:
            self._records = []
