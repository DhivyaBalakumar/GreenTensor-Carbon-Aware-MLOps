"""
GreenTensor Live Observability Dashboard
=========================================
Real-time carbon footprint + ML security monitoring.
Run with:  streamlit run dashboard/observability.py
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
    page_title="GreenTensor — Live Observability",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand styles ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .block-container { padding: 1.2rem 2rem 2rem; background: #0D1117; }
  section[data-testid="stSidebar"] { background: #161B22 !important; }
  section[data-testid="stSidebar"] * { color: #E6EDF3 !important; }

  /* Header */
  .gt-header {
    display: flex; align-items: center; gap: 12px;
    padding: 0.6rem 0 1rem;
    border-bottom: 1px solid #21262D;
    margin-bottom: 1.2rem;
  }
  .gt-logo { font-size: 1.6rem; }
  .gt-title { font-size: 1.4rem; font-weight: 800; color: #00C896; letter-spacing: -0.02em; }
  .gt-subtitle { font-size: 0.78rem; color: #8B949E; margin-top: 1px; }
  .gt-badge {
    margin-left: auto; background: #0D2137; border: 1px solid #00C896;
    color: #00C896; font-size: 0.7rem; font-weight: 700; padding: 3px 10px;
    border-radius: 20px; letter-spacing: 0.05em;
  }
  .gt-badge-live { background: #1A2E1A; border-color: #00C896; color: #00C896; }
  .gt-badge-paused { background: #2E1A1A; border-color: #FF5C5C; color: #FF5C5C; }

  /* KPI cards */
  .kpi-row { display: flex; gap: 12px; margin-bottom: 1.2rem; }
  .kpi-card {
    flex: 1; background: #161B22; border: 1px solid #21262D;
    border-radius: 8px; padding: 14px 16px;
  }
  .kpi-label { font-size: 0.68rem; font-weight: 600; color: #8B949E;
               text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
  .kpi-value { font-size: 1.8rem; font-weight: 800; color: #E6EDF3; line-height: 1; }
  .kpi-delta { font-size: 0.75rem; font-weight: 600; margin-top: 4px; }
  .kpi-green { color: #00C896; }
  .kpi-red   { color: #FF5C5C; }
  .kpi-amber { color: #F5A623; }
  .kpi-blue  { color: #4A9EFF; }

  /* Panel */
  .panel {
    background: #161B22; border: 1px solid #21262D;
    border-radius: 8px; padding: 16px 18px; margin-bottom: 12px;
  }
  .panel-title {
    font-size: 0.72rem; font-weight: 700; color: #8B949E;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px;
  }

  /* Verdict cards */
  .verdict-attack {
    background: #1A0A0A; border: 1.5px solid #FF5C5C;
    border-radius: 8px; padding: 14px 16px; margin: 8px 0;
  }
  .verdict-suspicious {
    background: #1A1200; border: 1.5px solid #F5A623;
    border-radius: 8px; padding: 14px 16px; margin: 8px 0;
  }
  .verdict-benign {
    background: #0A1A0A; border: 1.5px solid #00C896;
    border-radius: 8px; padding: 14px 16px; margin: 8px 0;
  }
  .verdict-title { font-size: 1.1rem; font-weight: 800; letter-spacing: 0.02em; }
  .verdict-attack   .verdict-title { color: #FF5C5C; }
  .verdict-suspicious .verdict-title { color: #F5A623; }
  .verdict-benign   .verdict-title { color: #00C896; }

  /* Score bar */
  .score-track { background: #21262D; border-radius: 4px; height: 8px; margin: 8px 0; overflow: hidden; }
  .score-fill  { height: 8px; border-radius: 4px; transition: width 0.3s ease; }

  /* Signal pills */
  .sig-active {
    display: inline-block; background: #2E1A00; border: 1px solid #F5A623;
    color: #F5A623; font-size: 0.68rem; font-weight: 700; padding: 2px 8px;
    border-radius: 4px; margin: 2px 3px 2px 0; letter-spacing: 0.04em;
  }
  .sig-clear {
    display: inline-block; background: #161B22; border: 1px solid #21262D;
    color: #8B949E; font-size: 0.68rem; padding: 2px 8px;
    border-radius: 4px; margin: 2px 3px 2px 0;
  }

  /* Alert banner */
  .alert-attack {
    background: #1A0A0A; border-left: 4px solid #FF5C5C;
    padding: 10px 14px; border-radius: 0 6px 6px 0; margin: 6px 0;
  }
  .alert-warn {
    background: #1A1200; border-left: 4px solid #F5A623;
    padding: 10px 14px; border-radius: 0 6px 6px 0; margin: 6px 0;
  }
  .alert-ok {
    background: #0A1A0A; border-left: 4px solid #00C896;
    padding: 10px 14px; border-radius: 0 6px 6px 0; margin: 6px 0;
  }
  .alert-text { font-size: 0.82rem; color: #E6EDF3; }
  .alert-action { font-size: 0.78rem; font-weight: 700; margin-top: 4px; }

  /* Evidence list */
  .evidence { font-size: 0.78rem; color: #8B949E; line-height: 1.7; }
  .evidence strong { color: #E6EDF3; }

  /* Mitre tag */
  .mitre-tag {
    display: inline-block; background: #0D2137; border: 1px solid #4A9EFF;
    color: #4A9EFF; font-size: 0.65rem; font-weight: 700; padding: 1px 7px;
    border-radius: 3px; margin-right: 4px; letter-spacing: 0.04em;
  }

  /* Divider */
  .gt-divider { border: none; border-top: 1px solid #21262D; margin: 1rem 0; }

  /* Streamlit overrides */
  .stButton > button {
    background: #00C896 !important; color: #0D1117 !important;
    font-weight: 700 !important; border: none !important;
    border-radius: 6px !important;
  }
  .stButton > button:hover { background: #00A87E !important; }
  [data-testid="stMetricValue"] { color: #E6EDF3 !important; }
  .stDataFrame { background: #161B22 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "power_history": [], "co2_history": [], "time_history": [],
    "signal_history": [], "match_results": [], "tick": 0,
    "attack_injected": False, "attack_type_injected": None,
    "alert_log": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

matcher = PatternMatcher()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 GreenTensor")
    st.markdown("**Live Observability Dashboard**")
    st.markdown("Carbon-Secure MLOps · v0.7.1")
    st.markdown("---")

    st.markdown("**Live Mode**")
    auto_refresh = st.toggle("Auto-refresh", value=False)
    refresh_rate = st.slider("Refresh interval (s)", 1, 5, 2)

    st.markdown("---")
    st.markdown("**Inject Attack Scenario**")
    st.caption("Simulate a real ML pipeline attack to see detection in action.")
    attack_scenario = st.selectbox("Attack type", [
        "None",
        "Cryptominer injection",
        "Data exfiltration",
        "Backdoor trigger",
        "Model theft / extraction",
    ])
    if st.button("Inject attack now", type="primary"):
        st.session_state.attack_injected = True
        st.session_state.attack_type_injected = attack_scenario

    if st.button("Clear attack / reset"):
        st.session_state.attack_injected = False
        st.session_state.attack_type_injected = None

    if st.button("Reset all data"):
        for k in defaults:
            st.session_state[k] = defaults[k] if not isinstance(defaults[k], list) else []
        st.rerun()

    st.markdown("---")
    st.markdown("**Detection Thresholds**")
    attack_threshold    = st.slider("Attack threshold",    50, 95, 80)
    suspicious_threshold = st.slider("Suspicious threshold", 20, 60, 50)

    st.markdown("---")
    st.markdown("**Webhook Alerts**")
    slack_url = st.text_input("Slack webhook URL (optional)", placeholder="https://hooks.slack.com/...")
    st.caption("Paste your Slack webhook to receive real alerts when attacks are detected.")


# ── Simulate one tick ─────────────────────────────────────────────────────────
def simulate_tick(tick, attack_injected, attack_type):
    base_power = 120.0 + 10 * math.sin(tick * 0.3)
    base_util  = 75.0  +  5 * math.sin(tick * 0.2)
    active_signals = []
    benign_context = []

    if attack_injected and attack_type and attack_type != "None":
        if "Cryptominer" in attack_type:
            power = base_power + random.uniform(80, 140)
            util  = random.uniform(85, 99)
            active_signals = ["power_spike_sustained", "new_process_spawned",
                              "high_gpu_util_unexpected", "no_training_activity"]
        elif "exfiltration" in attack_type:
            power = base_power + random.uniform(30, 60)
            util  = random.uniform(2, 8)
            active_signals = ["idle_drain", "network_outbound_unknown", "low_gpu_util_high_power"]
        elif "Backdoor" in attack_type:
            power = base_power + random.uniform(20, 50)
            util  = random.uniform(60, 80)
            active_signals = ["inference_latency_spike", "high_confidence_anomaly", "power_spike_brief"]
        elif "theft" in attack_type or "extraction" in attack_type:
            power = base_power + random.uniform(10, 30)
            util  = random.uniform(40, 70)
            active_signals = ["high_frequency_probing", "systematic_input_patterns",
                              "network_outbound_unknown"]
        else:
            power = base_power + random.uniform(-5, 5)
            util  = base_util
    else:
        power = base_power + random.uniform(-8, 8)
        util  = base_util  + random.uniform(-3, 3)
        if tick % 15 == 0:
            power += random.uniform(15, 25)
            benign_context = ["checkpoint_saving"]
        if tick % 20 == 0:
            power += random.uniform(10, 20)
            benign_context = ["evaluation_loop"]

    co2_rate = (power / 1000) * (1 / 3600) * 0.233
    return power, util, co2_rate, active_signals, benign_context


# ── Run tick ──────────────────────────────────────────────────────────────────
tick = st.session_state.tick
power, util, co2_rate, active_signals, benign_context = simulate_tick(
    tick, st.session_state.attack_injected, st.session_state.attack_type_injected or "None"
)
ts = datetime.now().strftime("%H:%M:%S")
st.session_state.power_history.append(power)
st.session_state.co2_history.append(co2_rate * 1e6)
st.session_state.time_history.append(ts)
st.session_state.signal_history.append(active_signals)
st.session_state.tick += 1

MAX = 60
for key in ["power_history", "co2_history", "time_history", "signal_history"]:
    if len(st.session_state[key]) > MAX:
        st.session_state[key] = st.session_state[key][-MAX:]

history = st.session_state.power_history
baseline_power = sum(history[-21:-1]) / max(len(history[-21:-1]), 1) if len(history) > 5 else power
delta_pct = ((power - baseline_power) / baseline_power * 100) if baseline_power > 0 else 0

# Pattern match
spike_detected = power > baseline_power * 1.3
match_result = None
if spike_detected and active_signals:
    match_result = matcher.match(
        power_w=power, baseline_power_w=baseline_power,
        gpu_util_pct=util, active_signals=active_signals,
        benign_context=benign_context,
    )
    if match_result.threat_score >= attack_threshold:
        match_result.verdict = "ATTACK"
    elif match_result.threat_score >= suspicious_threshold:
        match_result.verdict = "SUSPICIOUS"
    else:
        match_result.verdict = "BENIGN"

    st.session_state.match_results.append(match_result)
    if len(st.session_state.match_results) > 20:
        st.session_state.match_results = st.session_state.match_results[-20:]

    # Log alert
    if match_result.verdict in ("ATTACK", "SUSPICIOUS"):
        st.session_state.alert_log.append({
            "time": ts,
            "verdict": match_result.verdict,
            "score": match_result.threat_score,
            "type": match_result.attack_type or "unknown",
            "action": match_result.recommended_action,
        })
        if len(st.session_state.alert_log) > 50:
            st.session_state.alert_log = st.session_state.alert_log[-50:]

    # Fire Slack webhook if configured
    if slack_url and match_result.verdict == "ATTACK":
        try:
            import urllib.request, json as _json
            payload = _json.dumps({
                "text": (
                    f":rotating_light: *GreenTensor ATTACK DETECTED*\n"
                    f"*Type:* {match_result.attack_type}\n"
                    f"*Score:* {match_result.threat_score}/100\n"
                    f"*Power:* {power:.0f}W vs {baseline_power:.0f}W baseline ({delta_pct:+.1f}%)\n"
                    f"*Action:* {match_result.recommended_action}"
                )
            }).encode()
            req = urllib.request.Request(slack_url, data=payload,
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=3)
        except Exception:
            pass


# ── Header ────────────────────────────────────────────────────────────────────
mode_badge = "gt-badge-live" if auto_refresh else "gt-badge-paused"
mode_text  = "LIVE" if auto_refresh else "PAUSED"
st.markdown(f"""
<div class="gt-header">
  <span class="gt-logo">🌿</span>
  <div>
    <div class="gt-title">GreenTensor Observability</div>
    <div class="gt-subtitle">Carbon footprint + ML security · real-time monitoring</div>
  </div>
  <span class="gt-badge {mode_badge}">{mode_text}</span>
</div>
""", unsafe_allow_html=True)
st.caption(f"Last tick: {ts}  ·  Tick #{tick}  ·  Baseline power: {baseline_power:.0f}W")

# ── KPI row ───────────────────────────────────────────────────────────────────
total_alerts = len([r for r in st.session_state.match_results if r.verdict != "BENIGN"])
attack_count = len([r for r in st.session_state.match_results if r.verdict == "ATTACK"])
cumulative_co2_ug = sum(st.session_state.co2_history)

delta_color = "kpi-red" if delta_pct > 30 else "kpi-amber" if delta_pct > 10 else "kpi-green"
alert_color = "kpi-red" if total_alerts > 0 else "kpi-green"

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">GPU Power</div>
    <div class="kpi-value">{power:.0f}<span style="font-size:1rem;font-weight:500;color:#8B949E">W</span></div>
    <div class="kpi-delta {delta_color}">{delta_pct:+.1f}% vs baseline</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">GPU Utilization</div>
    <div class="kpi-value">{util:.0f}<span style="font-size:1rem;font-weight:500;color:#8B949E">%</span></div>
    <div class="kpi-delta kpi-blue">current</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">CO2 Rate</div>
    <div class="kpi-value">{co2_rate*1e6:.2f}<span style="font-size:0.9rem;font-weight:500;color:#8B949E"> µg/s</span></div>
    <div class="kpi-delta kpi-green">cumulative: {cumulative_co2_ug:.0f} µg</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Security Alerts</div>
    <div class="kpi-value {alert_color}">{total_alerts}</div>
    <div class="kpi-delta kpi-red">{attack_count} confirmed attack(s)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Active Signals</div>
    <div class="kpi-value {'kpi-red' if active_signals else 'kpi-green'}">{len(active_signals)}</div>
    <div class="kpi-delta {'kpi-amber' if active_signals else 'kpi-green'}">
      {'ANOMALY DETECTED' if active_signals else 'all clear'}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Main panels ───────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="medium")

with left:
    # Carbon timeline
    st.markdown('<div class="panel-title">Carbon Footprint Timeline</div>', unsafe_allow_html=True)
    df_power = pd.DataFrame({
        "Power (W)": st.session_state.power_history,
        "Baseline":  [baseline_power] * len(st.session_state.power_history),
    }, index=st.session_state.time_history)
    st.line_chart(df_power, height=200, use_container_width=True)

    st.markdown('<div class="panel-title" style="margin-top:12px">CO2 Emission Rate (µg/s)</div>',
                unsafe_allow_html=True)
    df_co2 = pd.DataFrame(
        {"CO2 (µg/s)": st.session_state.co2_history},
        index=st.session_state.time_history
    )
    st.line_chart(df_co2, height=130, use_container_width=True)

    # Spike / clean banner
    if spike_detected:
        st.markdown(f"""
        <div class="alert-warn">
          <div class="alert-text">
            <strong>POWER SPIKE DETECTED</strong> — {power:.0f}W ({delta_pct:+.1f}% above {baseline_power:.0f}W baseline)
          </div>
          <div class="alert-action" style="color:#F5A623">Pattern matching running...</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-ok">
          <div class="alert-text"><strong>NORMAL</strong> — Carbon footprint within expected range</div>
        </div>""", unsafe_allow_html=True)

    # Alert log
    if st.session_state.alert_log:
        st.markdown('<div class="panel-title" style="margin-top:16px">Alert Log</div>',
                    unsafe_allow_html=True)
        rows = []
        for a in reversed(st.session_state.alert_log[-8:]):
            rows.append({
                "Time": a["time"],
                "Verdict": a["verdict"],
                "Score": a["score"],
                "Type": a["type"],
                "Action": a["action"][:55] + "..." if len(a["action"]) > 55 else a["action"],
            })
        df_log = pd.DataFrame(rows)
        st.dataframe(df_log, use_container_width=True, hide_index=True)

with right:
    # Active signals
    st.markdown('<div class="panel-title">Security Signals</div>', unsafe_allow_html=True)
    all_signals = [
        "power_spike_sustained", "power_spike_brief", "idle_drain",
        "new_process_spawned", "high_gpu_util_unexpected", "no_training_activity",
        "network_outbound_unknown", "low_gpu_util_high_power",
        "inference_latency_spike", "high_confidence_anomaly",
        "high_frequency_probing", "systematic_input_patterns",
        "model_weight_modified", "file_access_anomaly",
    ]
    pills_html = ""
    for sig in all_signals:
        active = sig in active_signals
        css = "sig-active" if active else "sig-clear"
        pills_html += f'<span class="{css}">{sig.replace("_", " ")}</span>'
    st.markdown(f'<div style="line-height:2">{pills_html}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="gt-divider">', unsafe_allow_html=True)

    # Pattern match verdict
    st.markdown('<div class="panel-title">Pattern Match Verdict</div>', unsafe_allow_html=True)

    if match_result:
        score = match_result.threat_score
        score_color = "#FF5C5C" if score >= 80 else "#F5A623" if score >= 50 else "#00C896"
        css = f"verdict-{match_result.verdict.lower()}"

        # MITRE tags
        mitre_map = {
            "cryptominer":        ["AML.T0011"],
            "data_exfiltration":  ["AML.T0024"],
            "backdoor_trigger":   ["AML.T0043"],
            "model_theft":        ["AML.T0044"],
        }
        mitre_tags = "".join(
            f'<span class="mitre-tag">{t}</span>'
            for t in mitre_map.get(match_result.attack_type or "", [])
        )

        evidence_html = "<br>".join(
            f"&nbsp;&nbsp;• {e}" for e in match_result.evidence[:6]
        )

        st.markdown(f"""
        <div class="{css}">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div class="verdict-title">{match_result.verdict}</div>
            <div style="text-align:right">
              <div style="font-size:1.1rem;font-weight:800;color:{score_color}">{score}/100</div>
              <div style="font-size:0.68rem;color:#8B949E">threat score</div>
            </div>
          </div>
          <div class="score-track">
            <div class="score-fill" style="width:{score}%;background:{score_color}"></div>
          </div>
          <div style="margin:6px 0 4px">{mitre_tags}</div>
          <div style="font-size:0.8rem;color:#8B949E;margin-bottom:6px">
            <strong style="color:#E6EDF3">Type:</strong> {match_result.attack_type or "unknown"} &nbsp;|&nbsp;
            <strong style="color:#E6EDF3">Confidence:</strong> {match_result.confidence_pct:.0f}%
          </div>
          <div class="evidence">{evidence_html}</div>
          <div style="margin-top:8px;font-size:0.78rem;font-weight:700;color:{score_color}">
            {match_result.recommended_action}
          </div>
        </div>
        """, unsafe_allow_html=True)

    elif spike_detected:
        st.markdown("""
        <div class="verdict-suspicious">
          <div class="verdict-title">INVESTIGATING</div>
          <div style="font-size:0.82rem;color:#8B949E;margin-top:6px">
            Spike detected — no active attack signals. Likely benign (checkpoint, eval loop).
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="verdict-benign">
          <div class="verdict-title">BENIGN</div>
          <div style="font-size:0.82rem;color:#8B949E;margin-top:6px">
            No anomalies detected. System operating normally.
          </div>
        </div>""", unsafe_allow_html=True)

    # Recent match history
    if st.session_state.match_results:
        st.markdown('<div class="panel-title" style="margin-top:14px">Recent Verdicts</div>',
                    unsafe_allow_html=True)
        for r in reversed(st.session_state.match_results[-5:]):
            color = "#FF5C5C" if r.verdict == "ATTACK" else "#F5A623" if r.verdict == "SUSPICIOUS" else "#00C896"
            t = datetime.fromtimestamp(r.timestamp).strftime("%H:%M:%S")
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.78rem;padding:3px 0;border-bottom:1px solid #21262D">'
                f'<span style="color:#8B949E">{t}</span>'
                f'<span style="color:{color};font-weight:700">{r.verdict}</span>'
                f'<span style="color:#8B949E">{r.attack_type or "—"}</span>'
                f'<span style="color:{color}">{r.threat_score}/100</span>'
                f'</div>',
                unsafe_allow_html=True
            )

# ── Controls ──────────────────────────────────────────────────────────────────
st.markdown('<hr class="gt-divider">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if not auto_refresh:
        if st.button("Advance one tick"):
            st.rerun()
with col2:
    if st.button("Clear alerts"):
        st.session_state.match_results = []
        st.session_state.alert_log = []
        st.rerun()

if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
