# -*- coding: utf-8 -*-
"""
Pattern Matcher — correlates carbon footprint spikes with digital
footprint signals to determine if a spike is a cyberattack or benign.

Logic:
  When a carbon/power spike is detected, the matcher scores it against
  known attack patterns using weighted evidence from multiple signals.
  Each signal adds or subtracts from a threat score (0-100).
  Above 60 = likely attack. Above 80 = confirmed attack.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import time


@dataclass
class PatternMatchResult:
    timestamp: float
    threat_score: int           # 0-100
    verdict: str                # "BENIGN" | "SUSPICIOUS" | "ATTACK"
    attack_type: Optional[str]  # None | "cryptominer" | "data_exfiltration" | "backdoor" | "model_theft"
    confidence_pct: float
    evidence: List[str]         # human-readable evidence list
    recommended_action: str


class PatternMatcher:
    """
    Correlates carbon spike events with digital footprint signals.
    Runs instantly when a spike is detected.
    """

    # Attack pattern signatures — each is a dict of signal -> weight
    ATTACK_PATTERNS = {
        "cryptominer": {
            "power_spike_sustained": 35,
            "new_process_spawned": 25,
            "high_gpu_util_unexpected": 20,
            "no_training_activity": 15,
            "network_outbound_unknown": 5,
        },
        "data_exfiltration": {
            "idle_drain": 30,
            "network_outbound_unknown": 35,
            "low_gpu_util_high_power": 20,
            "file_access_anomaly": 15,
        },
        "backdoor_trigger": {
            "inference_latency_spike": 40,
            "high_confidence_anomaly": 30,
            "power_spike_brief": 20,
            "model_weight_modified": 10,
        },
        "model_theft": {
            "high_frequency_probing": 50,
            "systematic_input_patterns": 30,
            "network_outbound_unknown": 20,
        },
    }

    # Benign explanations that reduce threat score
    BENIGN_SIGNALS = {
        "batch_size_increased": -25,
        "new_layer_added": -20,
        "data_augmentation_active": -15,
        "checkpoint_saving": -20,
        "evaluation_loop": -10,
        "warmup_phase": -15,
    }

    def match(
        self,
        power_w: float,
        baseline_power_w: float,
        gpu_util_pct: float,
        active_signals: List[str],
        benign_context: Optional[List[str]] = None,
    ) -> PatternMatchResult:
        """
        Run pattern matching when a carbon spike is detected.

        Parameters
        ----------
        power_w          : current GPU power draw in watts
        baseline_power_w : rolling baseline power in watts
        gpu_util_pct     : current GPU utilization %
        active_signals   : list of signal names currently active
                           (from AnomalyDetector + DigitalFootprintScanner)
        benign_context   : optional list of known-benign context signals
        """
        deviation_pct = ((power_w - baseline_power_w) / baseline_power_w * 100) if baseline_power_w > 0 else 0
        evidence = []
        benign_context = benign_context or []

        # Score against each attack pattern
        pattern_scores = {}
        for attack_type, pattern in self.ATTACK_PATTERNS.items():
            score = 0
            matched = []
            for signal, weight in pattern.items():
                if signal in active_signals:
                    score += weight
                    matched.append(f"{signal} (+{weight})")
            if score > 0:
                pattern_scores[attack_type] = (score, matched)

        # Apply benign reductions
        benign_reduction = 0
        for ctx in benign_context:
            if ctx in self.BENIGN_SIGNALS:
                benign_reduction += abs(self.BENIGN_SIGNALS[ctx])
                evidence.append(f"Benign context: {ctx} (-{abs(self.BENIGN_SIGNALS[ctx])})")

        # Find the best matching attack pattern
        best_attack = None
        best_score = 0
        best_evidence = []
        for attack_type, (score, matched) in pattern_scores.items():
            adjusted = max(0, score - benign_reduction)
            if adjusted > best_score:
                best_score = adjusted
                best_attack = attack_type
                best_evidence = matched

        # Add power deviation to evidence
        evidence.insert(0, f"Power deviation: {deviation_pct:+.1f}% ({power_w:.1f}W vs {baseline_power_w:.1f}W baseline)")
        evidence.insert(1, f"GPU utilization: {gpu_util_pct:.1f}%")
        evidence += best_evidence

        # Determine verdict
        threat_score = min(100, best_score)
        if threat_score >= 80:
            verdict = "ATTACK"
            confidence = min(99.0, 60 + threat_score * 0.4)
            action = f"IMMEDIATE ACTION: Terminate pipeline. Suspected {best_attack}. Preserve logs for forensics."
        elif threat_score >= 50:
            verdict = "SUSPICIOUS"
            confidence = 40 + threat_score * 0.3
            action = f"INVESTIGATE: Monitor closely. Possible {best_attack}. Check process list and network connections."
        elif threat_score >= 20:
            verdict = "SUSPICIOUS"
            confidence = 20 + threat_score * 0.5
            action = "LOW RISK: Spike detected but insufficient evidence of attack. Continue monitoring."
        else:
            verdict = "BENIGN"
            best_attack = None
            confidence = max(0, 90 - threat_score * 2)
            action = "No action required. Spike consistent with normal training variation."

        return PatternMatchResult(
            timestamp=time.time(),
            threat_score=threat_score,
            verdict=verdict,
            attack_type=best_attack,
            confidence_pct=round(confidence, 1),
            evidence=evidence,
            recommended_action=action,
        )
