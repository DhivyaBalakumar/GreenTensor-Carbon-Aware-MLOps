# -*- coding: utf-8 -*-
"""
HuggingFace Transformers integration for GreenTensor.

Adds GreenTensor carbon tracking and security monitoring as a
HuggingFace TrainerCallback — zero changes to your training code.

Usage:
    from transformers import Trainer, TrainingArguments
    from greentensor.integrations.huggingface import GreenTensorCallback

    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks=[GreenTensorCallback(model_name="bert-finetuned")],
    )
    trainer.train()
"""

import time
from greentensor.utils.logger import logger

try:
    from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    TrainerCallback = object
    TrainerState = object
    TrainerControl = object
    TrainingArguments = object


class GreenTensorCallback(TrainerCallback):
    """
    HuggingFace TrainerCallback that wraps GreenTensor around
    the full training lifecycle.

    Tracks energy and carbon per epoch, detects security anomalies,
    and saves metrics after training completes.
    """

    def __init__(
        self,
        model_name: str = "hf_model",
        save_path: str = "greentensor_hf_metrics.pkl",
        security: bool = True,
        esg_reporter=None,
    ):
        if not HF_AVAILABLE:
            raise ImportError(
                "transformers is not installed. "
                "Install with: pip install transformers"
            )
        self.model_name = model_name
        self.save_path = save_path
        self.security = security
        self.esg_reporter = esg_reporter
        self._gt = None
        self._epoch_start = None
        self._epoch_metrics = []

    def on_train_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, **kwargs):
        from greentensor import GreenTensor
        self._gt = GreenTensor(
            security=self.security,
            save_path=self.save_path,
            verbose=False,
        )
        self._gt.__enter__()
        logger.info(f"GreenTensor tracking started for HuggingFace training: {self.model_name}")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._epoch_start:
            epoch_duration = time.perf_counter() - self._epoch_start
            self._epoch_metrics.append({
                "epoch": state.epoch,
                "duration_s": epoch_duration,
            })

    def on_train_end(self, args, state, control, **kwargs):
        if self._gt:
            self._gt.__exit__(None, None, None)
            metrics = self._gt.metrics
            alerts = self._gt.security_alerts

            logger.info(
                f"GreenTensor HF training complete: "
                f"{metrics.energy_kwh:.6f} kWh | "
                f"{metrics.emissions_kg:.6f} kg CO2 | "
                f"{len(alerts)} security alerts"
            )

            if self.esg_reporter and metrics:
                self.esg_reporter.record_run(
                    metrics=metrics,
                    model_name=self.model_name,
                    stage="fine_tuning",
                    security_incidents=len([a for a in alerts
                                           if getattr(a, 'severity', '') in ('high', 'critical')]),
                )

    def on_evaluate(self, args, state, control, **kwargs):
        """Pass benign context to anomaly detector during eval loops."""
        if self._gt and self._gt.anomaly_detector:
            pass  # eval loop is benign context — future: pass to pattern matcher
