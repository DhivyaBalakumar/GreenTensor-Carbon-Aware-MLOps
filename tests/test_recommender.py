from greentensor.report.metrics import RunMetrics
from greentensor.optimizers.recommender import EfficiencyRecommender


def _metrics(idle_s=0.0, duration_s=120.0, energy=0.001):
    return RunMetrics(duration_s=duration_s, energy_kwh=energy,
                      emissions_kg=0.0002, idle_seconds=idle_s)


def test_idle_time_recommendation():
    rec = EfficiencyRecommender()
    recs = rec.analyze(_metrics(idle_s=60.0, duration_s=120.0))
    assert any(r.category == "dataloader" for r in recs)
    assert any(r.priority == "high" for r in recs)


def test_no_recommendations_clean_run():
    rec = EfficiencyRecommender()
    recs = rec.analyze(
        _metrics(idle_s=0.0, duration_s=300.0),
        mixed_precision_enabled=True,
        batch_size=128,
        gpu_util_avg_pct=85.0,
    )
    assert not any(r.category == "dataloader" for r in recs)
    assert not any(r.category == "precision" for r in recs)


def test_mixed_precision_recommendation():
    rec = EfficiencyRecommender()
    recs = rec.analyze(_metrics(), mixed_precision_enabled=False)
    assert any(r.category == "precision" for r in recs)
    assert any(r.estimated_savings_pct > 20 for r in recs)


def test_small_batch_recommendation():
    rec = EfficiencyRecommender()
    recs = rec.analyze(_metrics(), batch_size=8)
    assert any(r.category == "batch_size" for r in recs)


def test_recommendations_sorted_by_priority():
    rec = EfficiencyRecommender()
    recs = rec.analyze(
        _metrics(idle_s=80.0, duration_s=100.0),
        mixed_precision_enabled=False,
        batch_size=4,
    )
    priorities = [r.priority for r in recs]
    # All high priority should come before medium
    seen_non_high = False
    for p in priorities:
        if p != "high":
            seen_non_high = True
        if seen_non_high and p == "high":
            assert False, "High priority recommendation after non-high"
