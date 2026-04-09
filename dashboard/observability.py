"""
GreenTensor Observability Dashboard

Live side-by-side view of:
  - Carbon footprint / GPU power timeline
  - Digital footprint / security signals
  - Pattern matching verdict when a spike is detected

Run with:
    streamlit run dashboard/observability.py
"""
import streamlit as st
import pandas as pd
import time
import random
import math
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from greentensor.security.pattern_matcher import PatternMatcher

st.set_page_config(
    page_title="GreenTensor Observability",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .obs-header { font-size: 1.4rem; font-weight: 800; color: #212529;
                  border-bottom: 3px solid #212529; padding-bottom: 0.5rem; margin-bottom: 1rem; }
    .panel { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px;
             padding: 1rem 1.2rem; margin-bottom: 0.8rem; }
    .panel-title { font-size: 0.75rem; font-weight: 700; color: #6c757d;
                   text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.4rem; }
    .big-number { font-size: 2rem; font-weight: 800; color: #212529; }
    .verdict-attack   { background:#fff0f0; border:2px solid #dc3545; border-radius:6px;
                        padding:1rem; margin:0.5rem 0; }
    .verdict-suspicious { background:#fff8f0; border:2px solid #fd7e14; border-radius:6px;
                          padding:1rem; margin:0.5rem 0; }
    .verdict-benign   { background:#f0fff4; border:2px solid #198754; border-radius:6px;
                        padding:1rem; margin:0.5rem 0; }
    .signal-active  { background:#fff3cd; border-left:3px solid #ffc107;
                      padding:0.3rem 0.6rem; margin:0.2rem 0; border-radius:3px;
                      font-size:0.82rem; }
    .signal-clear   { background:#d1e7dd; border-left:3px solid #198754;
                      padding:0.3rem 0.6rem; margin:0.2rem 0; border-radius:3px;
                      font-size:0.82rem; }
    .evidence-item  { font-size:0.82rem; color:#495057; padding:0.15rem 0; }
    .score-bar-bg   { background:#e9ecef; border-radius:4px; height:12px; margin:0.3rem 0; }
    .timeline-label { font-size:0.72rem; color:#6c757d; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "power_history" not in st.session_state:
    st.session_state.power_history = []
if "co2_history" not in st.session_state:
    st.session_state.co2_history = []
if "time_history" not in st.session_state:
    st.session_state.time_history = []
if "signal_history" not in st.session_state:
    st.session_state.signal_history = []
if "match_results" not in st.session_state:
    st.session_state.match_results = []
if "tick" not in st.session_state:
    st.session_state.tick = 0
if "attack_injected" not in st.session_state:
    st.session_state.attack_injected = False
if "attack_type_injected" not in st.session_state:
    st.session_state.attack_type_injected = None

matcher = PatternMatcher()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### GreenTensor Observability")
    st.markdown("Live carbon + security monitoring")
    st.markdown("---")

    st.markdown("**Simulation Controls**")
    auto_refresh = st.toggle("Live mode (auto-refresh)", value=False)
    refresh_rate = st.slider("Refresh interval (s)", 1, 5, 2)

    st.markdown("---")
    st.markdown("**Inject Attack Scenario**")
    attack_scenario = st.selectbox("Attack type", [
        "None",
        "Cryptominer injection",
        "Data exfiltration",
        "Backdoor trigger",
        "Model theft / extraction",
    ])
    if st.button("Inject now", type="primary"):
        st.session_state.attack_injected = True
        st.session_state.attack_type_injected = attack_scenario

    if st.button("Reset simulation"):
        for key in ["power_history", "co2_history", "time_history",
                    "signal_history", "match_results", "tick",
                    "attack_injected", "attack_type_injected"]:
            del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.markdown("**Pattern Matching Thresholds**")
    attack_threshold = st.slider("Attack threshold (score)", 50, 95, 80)
    suspicious_threshold = st.slider("Suspicious threshold (score)", 20, 60, 50)

    st.markdown("---")
    st.markdown("v0.4.0 | Dhivya Balakumar")


# ── Simulate one tick of data ─────────────────────────────────────────────────
def simulate_tick(tick: int, attack_injected: bool, attack_type: str):
    """Generate one second of simulated GPU power, CO2, and security signals."""
    base_power = 120.0 + 10 * math.sin(tick * 0.3)  # normal training oscillation
    base_util = 75.0 + 5 * math.sin(tick * 0.2)
    active_signals = []
    benign_context = []

    if attack_injected and attack_type != "None":
        if "Cryptominer" in attack_type:
            power = base_power + random.uniform(80, 140)
            util = random.uniform(85, 99)
            active_signals = ["power_spike_sustained", "new_process_spawned",
                              "high_gpu_util_unexpected", "no_training_activity"]
        elif "exfiltration" in attack_type:
            power = base_power + random.uniform(30, 60)
            util = random.uniform(2, 8)
            active_signals = ["idle_drain", "network_outbound_unknown",
                              "low_gpu_util_high_power"]
        elif "Backdoor" in attack_type:
            power = base_power + random.uniform(20, 50)
            util = random.uniform(60, 80)
            active_signals = ["inference_latency_spike", "high_confidence_anomaly",
                              "power_spike_brief"]
        elif "theft" in attack_type or "extraction" in attack_type:
            power = base_power + random.uniform(10, 30)
            util = random.uniform(40, 70)
            active_signals = ["high_frequency_probing", "systematic_input_patterns",
                              "network_outbound_unknown"]
        else:
            power = base_power + random.uniform(-5, 5)
            util = base_util
    else:
        power = base_power + random.uniform(-8, 8)
        util = base_util + random.uniform(-3, 3)
        # Occasional benign spikes
        if tick % 15 == 0:
            power += random.uniform(15, 25)
            benign_context = ["checkpoint_saving"]
        if tick % 20 == 0:
            power += random.uniform(10, 20)
            benign_context = ["evaluation_loop"]

    # CO2 = power * time * carbon_intensity (world avg 0.000233 kg/Wh)
    co2_rate = (power / 1000) * (1 / 3600) * 0.233  # kg per second

    return power, util, co2_rate, active_signals, benign_context


# ── Run one tick ──────────────────────────────────────────────────────────────
tick = st.session_state.tick
power, util, co2_rate, active_signals, benign_context = simulate_tick(
    tick,
    st.session_state.attack_injected,
    st.session_state.attack_type_injected or "None"
)

ts = datetime.now().strftime("%H:%M:%S")
st.session_state.power_history.append(power)
st.session_state.co2_history.append(co2_rate * 1e6)  # convert to µg/s for readability
st.session_state.time_history.append(ts)
st.session_state.signal_history.append(active_signals)
st.session_state.tick += 1

# Keep last 60 ticks
MAX = 60
for key in ["power_history", "co2_history", "time_history", "signal_history"]:
    if len(st.session_state[key]) > MAX:
        st.session_state[key] = st.session_state[key][-MAX:]

# Compute rolling baseline (last 20 ticks, excluding current)
history = st.session_state.power_history
baseline_power = sum(history[-21:-1]) / len(history[-21:-1]) if len(history) > 5 else power

# Run pattern matcher if spike detected
spike_detected = power > baseline_power * 1.3
match_result = None
if spike_detected and active_signals:
    match_result = matcher.match(
        power_w=power,
        baseline_power_w=baseline_power,
        gpu_util_pct=util,
        active_signals=active_signals,
        benign_context=benign_context,
    )
    # Override thresholds from sidebar
    if match_result.threat_score >= attack_threshold:
        match_result.verdict = "ATTACK"
    elif match_result.threat_score >= suspicious_threshold:
        match_result.verdict = "SUSPICIOUS"
    else:
        match_result.verdict = "BENIGN"

    st.session_state.match_results.append(match_result)
    if len(st.session_state.match_results) > 20:
        st.session_state.match_results = st.session_state.match_results[-20:]


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="obs-header">GreenTensor Observability Dashboard</div>', unsafe_allow_html=True)
st.caption(f"Last updated: {ts}  |  Tick: {tick}  |  Mode: {'LIVE' if auto_refresh else 'MANUAL'}")

# ── Top KPI row ───────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown('<div class="panel"><div class="panel-title">GPU Power</div>'
                f'<div class="big-number">{power:.0f}W</div></div>', unsafe_allow_html=True)
with k2:
    delta_pct = ((power - baseline_power) / baseline_power * 100) if baseline_power > 0 else 0
    color = "#dc3545" if delta_pct > 30 else "#198754"
    st.markdown(f'<div class="panel"><div class="panel-title">vs Baseline</div>'
                f'<div class="big-number" style="color:{color}">{delta_pct:+.1f}%</div></div>',
                unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="panel"><div class="panel-title">GPU Utilization</div>'
                f'<div class="big-number">{util:.0f}%</div></div>', unsafe_allow_html=True)
with k4:
    total_co2 = sum(st.session_state.co2_history)
    st.markdown(f'<div class="panel"><div class="panel-title">CO2 Rate (µg/s)</div>'
                f'<div class="big-number">{co2_rate*1e6:.2f}</div></div>', unsafe_allow_html=True)
with k5:
    total_alerts = len([r for r in st.session_state.match_results if r.verdict != "BENIGN"])
    alert_color = "#dc3545" if total_alerts > 0 else "#198754"
    st.markdown(f'<div class="panel"><div class="panel-title">Security Alerts</div>'
                f'<div class="big-number" style="color:{alert_color}">{total_alerts}</div></div>',
                unsafe_allow_html=True)

st.markdown("---")

# ── Main panels: Carbon | Security ───────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.markdown("**Carbon Footprint Timeline**")

    df_power = pd.DataFrame({
        "Time": st.session_state.time_history,
        "Power (W)": st.session_state.power_history,
        "Baseline": [baseline_power] * len(st.session_state.power_history),
    }).set_index("Time")
    st.line_chart(df_power, height=220)

    st.markdown("**CO2 Emission Rate (µg/s)**")
    df_co2 = pd.DataFrame({
        "Time": st.session_state.time_history,
        "CO2 rate (µg/s)": st.session_state.co2_history,
    }).set_index("Time")
    st.line_chart(df_co2, height=160)

    # Spike indicator
    if spike_detected:
        st.markdown(
            f'<div style="background:#fff3cd;border-left:4px solid #ffc107;'
            f'padding:0.6rem 1rem;border-radius:4px;margin-top:0.5rem;">'
            f'<strong>SPIKE DETECTED</strong> — {power:.0f}W ({delta_pct:+.1f}% above baseline) '
            f'— Pattern matching running...</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="background:#d1e7dd;border-left:4px solid #198754;'
            'padding:0.6rem 1rem;border-radius:4px;margin-top:0.5rem;">'
            'Carbon footprint within normal range</div>',
            unsafe_allow_html=True
        )

with right:
    st.markdown("**Active Security Signals**")

    all_signals = [
        "power_spike_sustained", "power_spike_brief", "idle_drain",
        "new_process_spawned", "high_gpu_util_unexpected", "no_training_activity",
        "network_outbound_unknown", "low_gpu_util_high_power", "file_access_anomaly",
        "inference_latency_spike", "high_confidence_anomaly", "model_weight_modified",
        "high_frequency_probing", "systematic_input_patterns",
    ]

    sig_col1, sig_col2 = st.columns(2)
    for i, sig in enumerate(all_signals):
        active = sig in active_signals
        css = "signal-active" if active else "signal-clear"
        icon = "ACTIVE" if active else "clear"
        col = sig_col1 if i % 2 == 0 else sig_col2
        with col:
            st.markdown(
                f'<div class="{css}">{sig.replace("_", " ")}<br>'
                f'<span style="font-size:0.7rem;font-weight:700;">{icon}</span></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown("**Pattern Match Result**")

    if match_result:
        css = f"verdict-{match_result.verdict.lower()}"
        score_color = "#dc3545" if match_result.threat_score >= 80 else \
                      "#fd7e14" if match_result.threat_score >= 50 else "#198754"

        st.markdown(f"""
        <div class="{css}">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:1.2rem;font-weight:800;">{match_result.verdict}</span>
                <span style="font-size:0.85rem;color:{score_color};font-weight:700;">
                    Threat score: {match_result.threat_score}/100
                </span>
            </div>
            <div class="score-bar-bg">
                <div style="width:{match_result.threat_score}%;background:{score_color};
                     height:12px;border-radius:4px;"></div>
            </div>
            <div style="font-size:0.85rem;margin-top:0.4rem;">
                <strong>Attack type:</strong> {match_result.attack_type or "None"}<br>
                <strong>Confidence:</strong> {match_result.confidence_pct:.1f}%
            </div>
            <div style="margin-top:0.6rem;font-size:0.82rem;color:#495057;">
                <strong>Evidence:</strong><br>
                {"<br>".join(f"&nbsp;&nbsp;• {e}" for e in match_result.evidence)}
            </div>
            <div style="margin-top:0.6rem;font-size:0.82rem;font-weight:600;">
                {match_result.recommended_action}
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif spike_detected:
        st.info("Spike detected but no active security signals — likely benign (checkpoint, eval loop).")
    else:
        st.markdown(
            '<div class="verdict-benign" style="padding:0.8rem;">'
            '<strong>BENIGN</strong> — No spike detected. System operating normally.</div>',
            unsafe_allow_html=True
        )

st.markdown("---")

# ── Alert history ─────────────────────────────────────────────────────────────
st.markdown("**Pattern Match History**")

if st.session_state.match_results:
    rows = []
    for r in reversed(st.session_state.match_results[-10:]):
        rows.append({
            "Time": datetime.fromtimestamp(r.timestamp).strftime("%H:%M:%S"),
            "Verdict": r.verdict,
            "Threat Score": r.threat_score,
            "Attack Type": r.attack_type or "—",
            "Confidence": f"{r.confidence_pct:.1f}%",
            "Action": r.recommended_action[:60] + "..." if len(r.recommended_action) > 60 else r.recommended_action,
        })
    df = pd.DataFrame(rows)

    def color_verdict(val):
        if val == "ATTACK":     return "background-color:#f8d7da;color:#58151c;font-weight:700"
        if val == "SUSPICIOUS": return "background-color:#fff3cd;color:#664d03;font-weight:700"
        return "background-color:#d1e7dd;color:#0a3622"

    st.dataframe(
        df.style.applymap(color_verdict, subset=["Verdict"]),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No pattern match events yet. Inject an attack scenario from the sidebar to see results.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
else:
    if st.button("Advance one tick"):
        st.rerun()
