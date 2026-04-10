# -*- coding: utf-8 -*-
"""
Model Efficiency Recommender for GreenTensor.

Analyzes telemetry from a completed run and produces actionable
recommendations to reduce energy consumption on the next run.
"""

from dataclasses import dataclass
from typing import List, Optional
from greentensor.report.metrics import RunMetrics


@dataclass
class Recommendation:
    category: str       # "dataloader" | "batch_size" | "precision" | "architecture" | "scheduling"
    priority: str       # "high" | "medium" | "low"
    title: str
    detail: str
    estimated_savings_pct: float


class EfficiencyRecommender:
    """
    Analyzes a RunMetrics object and produces ranked recommendations
    to reduce energy consumption on the next run.
    """

    def analyze(
        self,
        metrics: RunMetrics,
        gpu_util_avg_pct: Optional[float] = None,
        batch_size: Optional[int] = None,
        num_dataloader_workers: Optional[int] = None,
        mixed_precision_enabled: bool = True,
        model_params_millions: Optional[float] = None,
    ) -> List[Recommendation]:
        recs = []

        # 1. Idle time — data pipeline bottleneck
        if metrics.idle_seconds > 0 and metrics.duration_s > 0:
            idle_pct = metrics.idle_seconds / metrics.duration_s * 100
            if idle_pct > 20:
                recs.append(Recommendation(
                    category="dataloader",
                    priority="high",
                    title="GPU idle time is high — data pipeline bottleneck",
                    detail=(
                        f"GPU was idle {idle_pct:.1f}% of the run ({metrics.idle_seconds:.1f}s). "
                        f"Your DataLoader is likely CPU-bound. "
                        f"Add num_workers=4 (or more) to your DataLoader. "
                        f"Consider prefetch_factor=2 and pin_memory=True."
                    ),
                    estimated_savings_pct=min(idle_pct * 0.7, 35.0),
                ))
            elif idle_pct > 10:
                recs.append(Recommendation(
                    category="dataloader",
                    priority="medium",
                    title="Moderate GPU idle time detected",
                    detail=(
                        f"GPU was idle {idle_pct:.1f}% of the run. "
                        f"Consider increasing DataLoader workers or enabling data prefetching."
                    ),
                    estimated_savings_pct=idle_pct * 0.5,
                ))

        # 2. Mixed precision not enabled
        if not mixed_precision_enabled:
            recs.append(Recommendation(
                category="precision",
                priority="high",
                title="Mixed precision (FP16) not enabled",
                detail=(
                    "Enable mixed precision training with torch.cuda.amp.autocast(). "
                    "On Tensor Core GPUs (V100, A100, RTX 3000+), this reduces energy "
                    "by 20-40% with no accuracy loss. "
                    "Use GreenTensor's gt.mixed_precision() context manager."
                ),
                estimated_savings_pct=28.0,
            ))

        # 3. Batch size too small
        if batch_size and batch_size < 32:
            recs.append(Recommendation(
                category="batch_size",
                priority="high",
                title="Batch size is very small — GPU underutilized",
                detail=(
                    f"Current batch size: {batch_size}. "
                    f"Small batches cause high kernel launch overhead relative to compute. "
                    f"Use GreenTensor's optimize_batch_size() to find the optimal size "
                    f"based on available GPU memory. Typical recommendation: 64-256."
                ),
                estimated_savings_pct=15.0,
            ))
        elif batch_size and batch_size < 64:
            recs.append(Recommendation(
                category="batch_size",
                priority="medium",
                title="Batch size may be suboptimal",
                detail=(
                    f"Current batch size: {batch_size}. "
                    f"Consider increasing to 64+ if GPU memory allows. "
                    f"Run optimize_batch_size({batch_size}) to get a recommendation."
                ),
                estimated_savings_pct=8.0,
            ))

        # 4. GPU utilization low (if provided)
        if gpu_util_avg_pct is not None and gpu_util_avg_pct < 60:
            recs.append(Recommendation(
                category="scheduling",
                priority="medium",
                title="Average GPU utilization is low",
                detail=(
                    f"Average GPU utilization: {gpu_util_avg_pct:.1f}%. "
                    f"Below 60% suggests the GPU is waiting for data or compute is too small. "
                    f"Consider gradient accumulation to increase effective batch size, "
                    f"or profile your training loop for CPU bottlenecks."
                ),
                estimated_savings_pct=12.0,
            ))

        # 5. Large model — suggest distillation
        if model_params_millions and model_params_millions > 100:
            recs.append(Recommendation(
                category="architecture",
                priority="low",
                title="Large model — consider knowledge distillation",
                detail=(
                    f"Model has {model_params_millions:.0f}M parameters. "
                    f"For inference-heavy workloads, knowledge distillation to a smaller "
                    f"student model can reduce inference energy by 50-80% with <5% accuracy loss. "
                    f"Consider DistilBERT, TinyBERT, or task-specific pruning."
                ),
                estimated_savings_pct=40.0,
            ))

        # 6. Short run — overhead dominates
        if metrics.duration_s < 30 and metrics.energy_kwh > 0:
            recs.append(Recommendation(
                category="scheduling",
                priority="low",
                title="Very short run — consider batching experiments",
                detail=(
                    f"Run duration: {metrics.duration_s:.1f}s. "
                    f"Very short runs have high startup overhead relative to compute. "
                    f"Batch multiple experiments into a single job to amortize "
                    f"framework initialization and data loading costs."
                ),
                estimated_savings_pct=10.0,
            ))

        # Sort by priority then savings
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recs.sort(key=lambda r: (priority_order[r.priority], -r.estimated_savings_pct))
        return recs

    def print_recommendations(self, metrics: RunMetrics, **kwargs) -> List[Recommendation]:
        recs = self.analyze(metrics, **kwargs)
        if not recs:
            print("\n  All efficiency checks passed. No recommendations.\n")
            return recs
        print("\n  +-- Efficiency Recommendations -----------+")
        for i, r in enumerate(recs, 1):
            print(f"  {i}. [{r.priority.upper()}] {r.title}")
            print(f"     {r.detail}")
            print(f"     Estimated savings: {r.estimated_savings_pct:.0f}%")
            print()
        print("  +-----------------------------------------+\n")
        return recs
