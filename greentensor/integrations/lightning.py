# -*- coding: utf-8 -*-
"""
PyTorch Lightning integration for GreenTensor.

Adds GreenTensor as a Lightning Callback — zero changes to your
LightningModule or training code.

Usage:
    from pytorch_lightning import Trainer
    from greentensor.integrations.lightning import GreenTensorCallback

    trainer = Trainer(
        max_epochs=10,
        callbacks=[GreenTensorCallback(model_name="my_model")],
    )
    trainer.fit(model, datamodule)
"""

from greentensor.utils.logger import logger

try:
    from pytorch_lightning import Callback
    PL_AVAILABLE = True
except ImportError:
    try:
        from lightning import Callback
        PL_AVAILABLE = True
    except ImportError:
        PL_AVAILABLE = False
        Callback = object


class GreenTensorCallback(Callback):
    """
    PyTorch Lightning Callback that wraps GreenTensor around training.
    """

    def __init__(
        self,
        model_name: str = "lightning_model",
        save_path: str = "greentensor_lightning_metrics.pkl",
        security: bool = True,
        esg_reporter=None,
    ):
        if not PL_AVAILABLE:
            raise ImportError(
                "pytorch_lightning or lightning is not installed."
            )
        self.model_name = model_name
        self.save_path = save_path
        self.security = security
        self.esg_reporter = esg_reporter
        self._gt = None

    def on_fit_start(self, trainer, pl_module):
        from greentensor import GreenTensor
        self._gt = GreenTensor(
            security=self.security,
            save_path=self.save_path,
            verbose=False,
        )
        self._gt.__enter__()
        logger.info(f"GreenTensor tracking started for Lightning: {self.model_name}")

    def on_fit_end(self, trainer, pl_module):
        if self._gt:
            self._gt.__exit__(None, None, None)
            metrics = self._gt.metrics
            alerts = self._gt.security_alerts
            logger.info(
                f"GreenTensor Lightning complete: "
                f"{metrics.energy_kwh:.6f} kWh | "
                f"{metrics.emissions_kg:.6f} kg CO2 | "
                f"{len(alerts)} security alerts"
            )
            if self.esg_reporter and metrics:
                self.esg_reporter.record_run(
                    metrics=metrics,
                    model_name=self.model_name,
                    stage="training",
                    security_incidents=len([a for a in alerts
                                           if getattr(a, 'severity', '') in ('high', 'critical')]),
                )

    def on_validation_epoch_start(self, trainer, pl_module):
        """Validation is benign context for the anomaly detector."""
        pass

    def on_predict_end(self, trainer, pl_module):
        """Also track inference runs."""
        if self._gt:
            self._gt.__exit__(None, None, None)
