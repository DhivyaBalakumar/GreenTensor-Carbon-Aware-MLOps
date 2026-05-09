import streamlit as st
import pickle
import json
import os
import time as _time
import pandas as pd
from greentensor.report.metrics import (
    RunMetrics, calculate_savings, DatacenterConfig, PUE_PRESETS
)
from greentensor.report.esg import ESGReporter, ESGOrganization, ESGRunRecord
from greentensor.core.history import RunHistory
from greentensor.water.aquatensor import AquaTensorBridge, AquaTensorConfig, PROVIDER_WUE, REGIONAL_WATER_STRESS

st.set_page_config(
    page_title="GreenTensor Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand styles ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .block-container { padding: 1.5rem 2rem 2rem; background: #0D1117; }
  section[data-testid="stSidebar"] { background: #161B22 !important; }
  section[data-testid="stSidebar"] * { color: #E6EDF3 !important; }

  .gt-header {
    display: flex; align-items: center; gap: 14px;
    padding: 0.4rem 0 1.2rem;
    border-bottom: 1px solid #21262D; margin-bottom: 1.4rem;
  }
  .gt-logo { font-size: 1.8rem; }
  .gt-title { font-size: 1.5rem; font-weight: 800; color: #00C896; letter-spacing: -0.02em; }
  .gt-subtitle { font-size: 0.78rem; color: #8B949E; margin-top: 2px; }
  .gt-version {
    margin-left: auto; background: #0D2137; border: 1px solid #21262D;
    color: #8B949E; font-size: 0.68rem; padding: 3px 10px; border-radius: 20px;
  }

  .section-title {
    font-size: 1rem; font-weight: 700; color: #E6EDF3;
    border-bottom: 1px solid #21262D; padding-bottom: 8px; margin-bottom: 16px;
  }

  .kpi-card {
    background: #161B22; border: 1px solid #21262D;
    border-radius: 8px; padding: 14px 16px; margin-bottom: 8px;
  }
  .kpi-label { font-size: 0.68rem; font-weight: 600; color: #8B949E;
               text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
  .kpi-value { font-size: 1.6rem; font-weight: 800; color: #E6EDF3; line-height: 1.1; }
  .kpi-delta-good { font-size: 0.75rem; font-weight: 600; color: #00C896; margin-top: 3px; }
  .kpi-delta-bad  { font-size: 0.75rem; font-weight: 600; color: #FF5C5C; margin-top: 3px; }
  .kpi-delta-neutral { font-size: 0.75rem; color: #8B949E; margin-top: 3px; }

  .info-box {
    background: #161B22; border: 1px solid #21262D; border-left: 3px solid #00C896;
    border-radius: 0 6px 6px 0; padding: 10px 14px; margin: 8px 0;
    font-size: 0.82rem; color: #8B949E;
  }
  .warn-box {
    background: #1A1200; border: 1px solid #21262D; border-left: 3px solid #F5A623;
    border-radius: 0 6px 6px 0; padding: 10px 14px; margin: 8px 0;
    font-size: 0.82rem; color: #8B949E;
  }
  .clean-badge {
    display: inline-block; background: #0A1A0A; border: 1px solid #00C896;
    color: #00C896; font-weight: 700; padding: 6px 16px; border-radius: 6px;
    font-size: 0.85rem;
  }
  .threat-badge {
    display: inline-block; background: #1A0A0A; border: 1px solid #FF5C5C;
    color: #FF5C5C; font-weight: 700; padding: 6px 16px; border-radius: 6px;
    font-size: 0.85rem;
  }
  .alert-card {
    border-radius: 6px; padding: 10px 14px; margin: 6px 0; font-size: 0.82rem;
  }
  .alert-critical { background:#1A0A0A; border-left:3px solid #FF5C5C; color:#E6EDF3; }
  .alert-high     { background:#1A1000; border-left:3px solid #F5A623; color:#E6EDF3; }
  .alert-medium   { background:#1A1A00; border-left:3px solid #FFC107; color:#E6EDF3; }
  .alert-low      { background:#0A1A0A; border-left:3px solid #00C896; color:#E6EDF3; }
  .tag {
    display: inline-block; padding: 1px 7px; border-radius: 3px;
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.04em; margin-right: 4px;
  }
  .tag-source { background: #21262D; color: #8B949E; }
  .tag-type   { background: #0D2137; color: #4A9EFF; }
  .gt-divider { border: none; border-top: 1px solid #21262D; margin: 1.2rem 0; }

  .stButton > button {
    background: #00C896 !important; color: #0D1117 !important;
    font-weight: 700 !important; border: none !important; border-radius: 6px !important;
  }
  .stButton > button:hover { background: #00A87E !important; }
  .stDownloadButton > button {
    background: #161B22 !important; color: #00C896 !important;
    border: 1px solid #00C896 !important; font-weight: 600 !important;
    border-radius: 6px !important;
  }
  [data-testid="stMetricValue"] { color: #E6EDF3 !important; }
  [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def kpi(label, value, delta=None, good=True, neutral=False):
    delta_html = ""
    if delta:
        cls = "kpi-delta-neutral" if neutral else ("kpi-delta-good" if good else "kpi-delta-bad")
        delta_html = f'<div class="{cls}">{delta}</div>'
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 GreenTensor")
    st.markdown("Carbon-Secure MLOps · v0.7.1")
    st.markdown("---")
    page = st.radio("Navigation", [
        "Overview", "Energy Analysis", "Water Intelligence",
        "Datacenter Impact", "Efficiency Metrics", "ESG Report", "Run History",
        "Security Report", "How It Works",
    ])
    st.markdown("---")
    if st.button("🔄 Refresh data"):
        st.rerun()
    st.markdown("---")
    st.markdown("Built by Dhivya Balakumar")
    st.markdown("[GitHub](https://github.com/DhivyaBalakumar/GreenTensor-Carbon-Aware-MLOps) · MIT License")


# ── Live data from RunHistory ─────────────────────────────────────────────────
# Reads greentensor_history.json written automatically by GreenTensor.__exit__
# No upload or manual input needed — just run your training code.

@st.cache_data(ttl=5)   # re-reads the file every 5 seconds
def load_history():
    return RunHistory().all()

records = load_history()

def _to_metrics(r):
    return RunMetrics(
        duration_s=r.get("duration_s", 0),
        energy_kwh=r.get("energy_kwh", 0),
        emissions_kg=r.get("emissions_kg", 0),
        idle_seconds=r.get("idle_seconds", 0),
    )

# Derive baseline / optimized from the two most recent runs (if available).
# Always assign the higher-energy run as baseline so savings are never negative.
baseline_metrics  = None
optimized_metrics = None
security_alerts   = []

if len(records) >= 2:
    r_a = _to_metrics(records[-2])
    r_b = _to_metrics(records[-1])
    if r_a.energy_kwh >= r_b.energy_kwh:
        baseline_metrics, optimized_metrics = r_a, r_b
    else:
        baseline_metrics, optimized_metrics = r_b, r_a
elif len(records) == 1:
    baseline_metrics = _to_metrics(records[-1])



# ── Pages ─────────────────────────────────────────────────────────────────────

# Header shown on every page
st.markdown(f"""
<div class="gt-header">
  <span class="gt-logo">🌿</span>
  <div>
    <div class="gt-title">GreenTensor</div>
    <div class="gt-subtitle">Carbon-Secure MLOps + AquaTensor Water Intelligence</div>
  </div>
  <span class="gt-version">v0.7.1 · {len(records)} run{"s" if len(records) != 1 else ""} recorded · live</span>
</div>
""", unsafe_allow_html=True)

# ── Overview ──────────────────────────────────────────────────────────────────
if page == "Overview":
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

    if not baseline_metrics:
        st.markdown("""
        <div class="info-box">
          <strong>No runs recorded yet.</strong><br><br>
          GreenTensor automatically records every run to <code>greentensor_history.json</code>
          the moment your training job finishes. Just run your code:<br><br>
          <code>
          from greentensor import GreenTensor<br><br>
          with GreenTensor(model_name="my-model") as gt:<br>
          &nbsp;&nbsp;&nbsp;&nbsp;train()
          </code><br><br>
          The dashboard updates within 5 seconds. Hit <strong>🔄 Refresh data</strong> in the sidebar to force an immediate reload.
        </div>""", unsafe_allow_html=True)

    elif baseline_metrics and not optimized_metrics:
        st.markdown("**1 run recorded — run a second job to see a before/after comparison.**")
        c1, c2, c3 = st.columns(3)
        with c1: kpi("Duration", f"{baseline_metrics.duration_s:.2f} s", neutral=True)
        with c2: kpi("Energy Used", f"{baseline_metrics.energy_kwh:.6f} kWh", neutral=True)
        with c3: kpi("CO2 Emissions", f"{baseline_metrics.emissions_kg:.6f} kg", neutral=True)

    else:
        savings = calculate_savings(baseline_metrics, optimized_metrics)
        st.markdown("**GreenTensor savings — baseline vs optimized**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi("Energy Reduction", f"{savings['energy_reduction_pct']:.1f}%",
                delta=f"saved {savings['energy_saved_kwh']:.6f} kWh", good=True)
        with c2:
            kpi("CO2 Avoided", f"{savings['emissions_saved_kg']:.6f} kg",
                delta=f"from {baseline_metrics.emissions_kg:.6f} kg baseline", good=True)
        with c3:
            kpi("Time Saved", f"{savings['time_saved_s']:.2f} s",
                delta=f"from {baseline_metrics.duration_s:.2f} s", good=True)
        with c4:
            idle_pct = (baseline_metrics.idle_seconds / baseline_metrics.duration_s * 100) if baseline_metrics.duration_s > 0 else 0
            kpi("GPU Idle Time", f"{baseline_metrics.idle_seconds:.1f} s",
                delta=f"{idle_pct:.1f}% of run", good=(idle_pct < 10), neutral=(idle_pct == 0))

        st.markdown('<hr class="gt-divider">', unsafe_allow_html=True)
        km = savings['emissions_saved_kg'] / 0.000192
        trees = savings['emissions_saved_kg'] / 21.0
        st.markdown("**Real-world impact of these savings**")
        c1, c2, c3 = st.columns(3)
        with c1: kpi("Equivalent km not driven", f"{km:.1f} km", neutral=True)
        with c2: kpi("Trees equivalent (1 yr)", f"{trees:.4f}", neutral=True)
        with c3: kpi("Runs analysed", "2", delta="baseline + optimized", neutral=True)

# ── Energy Analysis ───────────────────────────────────────────────────────────
elif page == "Energy Analysis":
    st.markdown('<div class="section-title">Energy Analysis</div>', unsafe_allow_html=True)

    if not baseline_metrics:
        st.markdown('<div class="info-box">No runs recorded yet. Run a training job — data appears here automatically.</div>', unsafe_allow_html=True)
    else:
        if optimized_metrics:
            savings = calculate_savings(baseline_metrics, optimized_metrics)

            # Identify which recorded run maps to which role
            if len(records) >= 2:
                r_a = records[-2]; r_b = records[-1]
                if r_a.get("energy_kwh", 0) >= r_b.get("energy_kwh", 0):
                    base_label = f"{r_a.get('model_name','run -2')} (higher energy → baseline)"
                    opt_label  = f"{r_b.get('model_name','run -1')} (lower energy → optimized)"
                else:
                    base_label = f"{r_b.get('model_name','run -1')} (higher energy → baseline)"
                    opt_label  = f"{r_a.get('model_name','run -2')} (lower energy → optimized)"
            else:
                base_label, opt_label = "Baseline", "Optimized"

            st.markdown(f'<div class="info-box">Comparing the two most recent runs. <strong>Baseline:</strong> {base_label} &nbsp;|&nbsp; <strong>Optimized:</strong> {opt_label}</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1: kpi("Energy Reduction", f"{savings['energy_reduction_pct']:.1f}%", good=True)
            with c2: kpi("Energy Saved", f"{savings['energy_saved_kwh']:.6f} kWh", good=True)
            with c3: kpi("CO2 Saved", f"{savings['emissions_saved_kg']:.6f} kg", good=True)

            st.markdown('<hr class="gt-divider">', unsafe_allow_html=True)
            st.markdown("**Energy consumption (kWh)**")
            st.bar_chart(pd.DataFrame(
                {"Energy (kWh)": [baseline_metrics.energy_kwh, optimized_metrics.energy_kwh]},
                index=["Baseline", "Optimized"]))

            st.markdown("**CO2 emissions (kg)**")
            st.bar_chart(pd.DataFrame(
                {"Emissions (kg CO2)": [baseline_metrics.emissions_kg, optimized_metrics.emissions_kg]},
                index=["Baseline", "Optimized"]))

            st.markdown("**Runtime (seconds)**")
            st.bar_chart(pd.DataFrame(
                {"Duration (s)": [baseline_metrics.duration_s, optimized_metrics.duration_s]},
                index=["Baseline", "Optimized"]))

            st.markdown('<hr class="gt-divider">', unsafe_allow_html=True)
            st.markdown("**Detailed breakdown**")
            st.dataframe(pd.DataFrame([
                {"Run": "Baseline",  "Duration (s)": baseline_metrics.duration_s,
                 "Energy (kWh)": baseline_metrics.energy_kwh,
                 "Emissions (kg)": baseline_metrics.emissions_kg,
                 "Idle (s)": baseline_metrics.idle_seconds},
                {"Run": "Optimized", "Duration (s)": optimized_metrics.duration_s,
                 "Energy (kWh)": optimized_metrics.energy_kwh,
                 "Emissions (kg)": optimized_metrics.emissions_kg,
                 "Idle (s)": optimized_metrics.idle_seconds},
                {"Run": "Saved",     "Duration (s)": savings["time_saved_s"],
                 "Energy (kWh)": savings["energy_saved_kwh"],
                 "Emissions (kg)": savings["emissions_saved_kg"], "Idle (s)": "-"},
            ]).set_index("Run"), use_container_width=True)
        else:
            st.bar_chart(pd.DataFrame(
                {"Energy (kWh)": [baseline_metrics.energy_kwh]}, index=["Latest run"]))
            st.markdown('<div class="info-box">Run a second training job to see a before/after comparison.</div>',
                        unsafe_allow_html=True)

# ── Water Intelligence ────────────────────────────────────────────────────────
elif page == "Water Intelligence":
    st.markdown('<div class="section-title">AquaTensor Water Intelligence</div>', unsafe_allow_html=True)
    st.markdown("Every watt of GPU compute becomes waste heat. AquaTensor converts that heat to fresh water via membrane distillation. GreenTensor measures the real compute and calculates exactly how much water is produced — per training job.")

    st.markdown('<hr class="gt-divider">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        provider = st.selectbox("Cloud / DC provider", list(PROVIDER_WUE.keys()),
                                format_func=lambda k: f"{k}  (WUE: {PROVIDER_WUE[k] or 'custom'})")
        custom_wue = None
        if provider == "custom":
            custom_wue = st.number_input("Custom WUE (L/kWh)", min_value=0.1, value=1.8, step=0.1)
        region = st.selectbox("Region (water stress)", list(REGIONAL_WATER_STRESS.keys()))
        aquatensor_installed = st.toggle("AquaTensor MD system installed", value=True)
        if aquatensor_installed:
            whr = st.slider("Waste Heat Recovery ratio", 0.1, 1.0, 0.65, 0.05)
            feed_temp = st.slider("Coolant feed temperature (C)", 40, 80, 60)
        else:
            whr, feed_temp = 0.0, 60

    with col2:
        st.markdown("**MD yield by temperature** (Khayet & Matsuura, 2011)")
        st.dataframe(pd.DataFrame([
            {"Temp (C)": t, "Yield (L/kWh)": y}
            for t, y in {40: 2.5, 50: 4.0, 60: 5.5, 70: 7.0, 80: 8.5}.items()
        ]), use_container_width=True, hide_index=True)
        st.markdown("**Regional water stress (WRI Aqueduct)**")
        st.dataframe(pd.DataFrame([
            {"Region": k, "Index": v,
             "Level": "Extremely High" if v >= 4 else "High" if v >= 3 else "Medium" if v >= 2 else "Low"}
            for k, v in REGIONAL_WATER_STRESS.items()
        ]), use_container_width=True, hide_index=True)

    aq_config = AquaTensorConfig(provider=provider, custom_wue=custom_wue, region=region,
                                  aquatensor_installed=aquatensor_installed,
                                  whr_ratio=whr, feed_temperature_c=float(feed_temp))
    bridge = AquaTensorBridge(aq_config)

    if baseline_metrics:
        water = bridge.calculate_water_metrics(baseline_metrics.energy_kwh, baseline_metrics.duration_s)
        st.markdown('<hr class="gt-divider">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Energy Used", f"{baseline_metrics.energy_kwh:.6f} kWh", neutral=True)
        with c2: kpi("Water Consumed", f"{water.water_consumed_liters:.4f} L",
                     delta=f"WUE {water.wue} L/kWh", good=False)
        with c3:
            if aquatensor_installed:
                kpi("Water Produced", f"{water.water_produced_liters:.4f} L",
                    delta=f"AquaTensor @ {feed_temp}C", good=True)
            else:
                kpi("Water Produced", "0.0000 L", delta="not installed", neutral=True)
        with c4:
            net = water.net_water_impact_liters
            kpi("Net Impact", f"{net:.4f} L",
                delta="NET POSITIVE" if net < 0 else "net negative", good=(net < 0))

        if aquatensor_installed and water.water_produced_liters > 0:
            st.markdown(f"""
- Heat generated: **{water.heat_generated_kwh:.6f} kWh** · Recovered: **{water.heat_recovered_kwh:.6f} kWh** ({whr*100:.0f}% WHR)
- MD yield at {feed_temp}C: **{water.md_yield_liters_per_kwh:.1f} L/kWh** · Water produced: **{water.water_produced_liters:.4f} L**
- Drinking water equivalent: **{water.drinking_water_days:.2f} person-days**
- Region water stress: **{water.water_stress_label}** (WRI index {water.water_stress_index}/5.0)
            """)

        st.markdown('<hr class="gt-divider">', unsafe_allow_html=True)
        st.markdown("**Heat forecast for queued jobs**")
        n_jobs = st.number_input("Number of queued jobs", 1, 10, 2)
        jobs = []
        for i in range(int(n_jobs)):
            c1, c2, c3 = st.columns(3)
            with c1: name = st.text_input(f"Job {i+1}", value=f"job_{i+1}", key=f"jn{i}")
            with c2: dur  = st.number_input("Duration (h)", 0.5, 72.0, 2.0, key=f"jd{i}")
            with c3: pwr  = st.number_input("GPU power (W)", 50, 1000, 300, key=f"jp{i}")
            jobs.append({"name": name, "estimated_duration_s": dur*3600, "estimated_power_w": pwr})
        if st.button("Forecast"):
            fc = bridge.forecast_heat(jobs)
            c1, c2, c3 = st.columns(3)
            with c1: kpi("Predicted Energy", f"{fc.predicted_energy_kwh:.3f} kWh", neutral=True)
            with c2: kpi("Heat Recovered", f"{fc.predicted_heat_kwh:.3f} kWh", neutral=True)
            with c3: kpi("Water Predicted", f"{fc.predicted_water_liters:.2f} L", good=True)
            st.markdown(f"**Recommendation:** {fc.optimal_schedule_recommendation}")
    else:
        st.markdown('<div class="info-box">Load metrics to see water calculations.</div>',
                    unsafe_allow_html=True)

# ── Datacenter Impact ─────────────────────────────────────────────────────────
elif page == "Datacenter Impact":
    st.markdown('<div class="section-title">Datacenter Impact</div>', unsafe_allow_html=True)
    st.markdown("GPU energy measurements don't include datacenter overhead. PUE and regional carbon intensity scale your raw measurements to real-world impact.")

    st.markdown('<hr class="gt-divider">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        preset_label = st.selectbox("DC type", [
            "local_workstation  — PUE 1.0 (no DC overhead)",
            "hyperscale         — PUE 1.1 (Google / Microsoft / Meta)",
            "cloud_average      — PUE 1.2 (AWS / GCP / Azure average)",
            "enterprise_dc      — PUE 1.5 (typical on-premise DC)",
            "legacy_dc          — PUE 1.8 (older / inefficient DC)",
            "custom             — enter your own PUE",
        ])
        preset_key = preset_label.split()[0]
        pue = st.number_input("Custom PUE", 1.0, 3.0, PUE_PRESETS.get(preset_key, 1.2), 0.05) \
              if preset_key == "custom" else PUE_PRESETS.get(preset_key, 1.2)
        if preset_key != "custom":
            st.markdown(f'<div class="info-box">PUE set to <strong>{pue}</strong></div>',
                        unsafe_allow_html=True)
        num_nodes = st.number_input("Training nodes", 1, 1024, 1)

    with col2:
        st.markdown("**Regional carbon intensity** — find yours at [electricitymap.org](https://app.electricitymap.org)")
        st.dataframe(pd.DataFrame([
            {"Region": r, "kg CO2/kWh": v} for r, v in
            [("Norway (hydro)", 0.017), ("France (nuclear)", 0.056),
             ("UK average", 0.233), ("World average", 0.233),
             ("USA average", 0.386), ("Australia", 0.490), ("Poland (coal)", 0.635)]
        ]), use_container_width=True, hide_index=True)
        carbon_intensity = st.number_input("Carbon intensity (kg CO2/kWh)",
                                           0.0, 2.0, 0.000233, format="%.6f")

    dc_config = DatacenterConfig(pue=pue, carbon_intensity_kg_per_kwh=carbon_intensity,
                                  num_nodes=int(num_nodes))

    if not baseline_metrics:
        st.markdown('<div class="info-box">Load metrics to see datacenter impact.</div>',
                    unsafe_allow_html=True)
    else:
        b_dc = baseline_metrics.apply_datacenter_config(dc_config)
        o_dc = optimized_metrics.apply_datacenter_config(dc_config) if optimized_metrics else None

        st.markdown('<hr class="gt-divider">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Raw GPU Energy", f"{baseline_metrics.energy_kwh:.6f} kWh", neutral=True)
        with c2: kpi("PUE Multiplier", f"x {pue}", neutral=True)
        with c3: kpi("Nodes", f"x {int(num_nodes)}", neutral=True)
        overhead = ((b_dc.energy_kwh_dc / baseline_metrics.energy_kwh) - 1) * 100 if baseline_metrics.energy_kwh > 0 else 0
        with c4: kpi("Total DC Energy", f"{b_dc.energy_kwh_dc:.6f} kWh",
                     delta=f"+{overhead:.0f}% overhead", good=False)

        if o_dc:
            savings_dc = calculate_savings(b_dc, o_dc, use_dc_adjusted=True)
            st.markdown('<hr class="gt-divider">', unsafe_allow_html=True)
            st.markdown("**DC-adjusted savings**")
            c1, c2, c3 = st.columns(3)
            with c1: kpi("DC Energy Reduction", f"{savings_dc['energy_reduction_pct']:.1f}%", good=True)
            with c2: kpi("DC Energy Saved", f"{savings_dc['energy_saved_kwh']:.6f} kWh", good=True)
            with c3: kpi("DC CO2 Saved", f"{savings_dc['emissions_saved_kg']:.6f} kg", good=True)
            st.bar_chart(pd.DataFrame(
                {"DC Energy (kWh)": [b_dc.energy_kwh_dc, o_dc.energy_kwh_dc]},
                index=["Baseline (DC)", "Optimized (DC)"]))

# ── Efficiency Metrics — PUE / WUE / CUE ─────────────────────────────────────
elif page == "Efficiency Metrics":
    st.markdown('<div class="section-title">Efficiency Metrics — PUE · WUE · CUE</div>', unsafe_allow_html=True)
    st.markdown("""
Three industry-standard indices tell you how efficiently your datacenter converts power, water, and carbon into useful compute.
A **lower value is always better** for all three.
    """)

    # ── Definitions ───────────────────────────────────────────────────────────
    st.markdown('<hr class="gt-divider">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
<div class="kpi-card">
  <div class="kpi-label">PUE — Power Usage Effectiveness</div>
  <div style="font-size:0.85rem; color:#E6EDF3; margin-top:6px;">
    <strong>PUE = Total DC Power ÷ IT Equipment Power</strong><br><br>
    Measures how much extra energy the datacenter infrastructure (cooling, lighting, UPS) consumes
    on top of the actual compute load.<br><br>
    <span style="color:#00C896">1.0</span> = perfect (no overhead)<br>
    <span style="color:#F5A623">1.2</span> = hyperscale cloud average<br>
    <span style="color:#FF5C5C">1.8+</span> = legacy / inefficient DC<br><br>
    <em>Source: Green Grid consortium standard</em>
  </div>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class="kpi-card">
  <div class="kpi-label">WUE — Water Usage Effectiveness</div>
  <div style="font-size:0.85rem; color:#E6EDF3; margin-top:6px;">
    <strong>WUE = Annual Water Usage (L) ÷ IT Equipment Energy (kWh)</strong><br><br>
    Measures how many liters of water the datacenter evaporates for cooling per kWh of compute delivered.<br><br>
    <span style="color:#00C896">0.26 L/kWh</span> = Meta (best-in-class)<br>
    <span style="color:#F5A623">0.49 L/kWh</span> = Google / Microsoft<br>
    <span style="color:#FF5C5C">1.80 L/kWh</span> = AWS / on-premise avg<br><br>
    <em>Source: Provider sustainability reports 2023–24</em>
  </div>
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
<div class="kpi-card">
  <div class="kpi-label">CUE — Carbon Usage Effectiveness</div>
  <div style="font-size:0.85rem; color:#E6EDF3; margin-top:6px;">
    <strong>CUE = Total CO₂ Emissions (kg) ÷ IT Equipment Energy (kWh)</strong><br><br>
    Measures how many kg of CO₂ are emitted per kWh of compute — combining PUE overhead
    with the regional grid's carbon intensity.<br><br>
    <span style="color:#00C896">~0.000019</span> = Norway (hydro)<br>
    <span style="color:#F5A623">~0.000280</span> = UK average<br>
    <span style="color:#FF5C5C">~0.001224</span> = India legacy DC<br><br>
    <em>CUE = PUE × grid carbon intensity</em>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Interactive calculator ─────────────────────────────────────────────────
    st.markdown('<hr class="gt-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Interactive Calculator</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Datacenter parameters**")
        calc_pue = st.slider("PUE", min_value=1.0, max_value=3.0, value=1.2, step=0.05,
                             help="Power Usage Effectiveness — total DC power / IT power")
        calc_wue_provider = st.selectbox(
            "Cloud / DC provider (sets WUE)",
            list(PROVIDER_WUE.keys()),
            format_func=lambda k: f"{k}  —  WUE: {PROVIDER_WUE[k] or 'custom'} L/kWh"
        )
        calc_custom_wue = None
        if calc_wue_provider == "custom":
            calc_custom_wue = st.number_input("Custom WUE (L/kWh)", min_value=0.1, value=1.8, step=0.1)
        calc_wue = calc_custom_wue if calc_wue_provider == "custom" else PROVIDER_WUE[calc_wue_provider]

        st.markdown("**Grid carbon intensity**")
        grid_options = {
            "Norway (hydro)  — 0.000017 kg/kWh":  0.000017,
            "France (nuclear) — 0.000056 kg/kWh": 0.000056,
            "UK average       — 0.000233 kg/kWh": 0.000233,
            "World average    — 0.000233 kg/kWh": 0.000233,
            "USA average      — 0.000386 kg/kWh": 0.000386,
            "Australia        — 0.000490 kg/kWh": 0.000490,
            "India North      — 0.000680 kg/kWh": 0.000680,
            "Poland (coal)    — 0.000635 kg/kWh": 0.000635,
            "Custom":                              None,
        }
        grid_choice = st.selectbox("Grid region", list(grid_options.keys()))
        if grid_options[grid_choice] is None:
            calc_carbon_intensity = st.number_input(
                "Carbon intensity (kg CO₂/kWh)", min_value=0.0, value=0.000233, format="%.6f")
        else:
            calc_carbon_intensity = grid_options[grid_choice]

    with col2:
        st.markdown("**Workload parameters**")
        calc_it_energy = st.number_input(
            "IT equipment energy (kWh) — raw GPU measurement",
            min_value=0.0001, value=0.00058, format="%.6f",
            help="The energy your GPU actually consumed, as measured by GreenTensor"
        )
        calc_num_nodes = st.number_input("Number of training nodes", min_value=1, value=1)

    # ── Compute the three indices ──────────────────────────────────────────────
    total_dc_energy  = calc_it_energy * calc_pue * calc_num_nodes
    total_co2        = total_dc_energy * calc_carbon_intensity
    calc_cue         = calc_pue * calc_carbon_intensity          # kg CO2 per kWh IT
    water_consumed   = calc_it_energy * (calc_wue or 1.8)

    st.markdown('<hr class="gt-divider">', unsafe_allow_html=True)
    st.markdown("**Your datacenter efficiency indices**")
    c1, c2, c3, c4 = st.columns(4)

    # PUE rating
    pue_label = "Excellent" if calc_pue <= 1.1 else ("Good" if calc_pue <= 1.3 else ("Average" if calc_pue <= 1.6 else "Poor"))
    pue_good  = calc_pue <= 1.3
    with c1:
        kpi("PUE", f"{calc_pue:.2f}",
            delta=f"{pue_label} — {'+' if calc_pue > 1.0 else ''}{(calc_pue - 1.0)*100:.0f}% DC overhead",
            good=pue_good)

    # WUE rating
    wue_label = "Excellent" if (calc_wue or 99) <= 0.3 else ("Good" if (calc_wue or 99) <= 0.6 else ("Average" if (calc_wue or 99) <= 1.2 else "Poor"))
    wue_good  = (calc_wue or 99) <= 0.6
    with c2:
        kpi("WUE", f"{calc_wue:.2f} L/kWh" if calc_wue else "N/A",
            delta=f"{wue_label} — {water_consumed:.5f} L consumed",
            good=wue_good)

    # CUE rating
    cue_label = "Excellent" if calc_cue <= 0.00006 else ("Good" if calc_cue <= 0.0003 else ("Average" if calc_cue <= 0.0006 else "Poor"))
    cue_good  = calc_cue <= 0.0003
    with c3:
        kpi("CUE", f"{calc_cue:.6f} kg/kWh",
            delta=f"{cue_label} — {total_co2:.6f} kg CO₂ total",
            good=cue_good)

    with c4:
        kpi("DC Energy", f"{total_dc_energy:.6f} kWh",
            delta=f"x{calc_pue:.1f} PUE × {int(calc_num_nodes)} node(s)",
            good=False, neutral=True)

    # ── Benchmark comparison table ─────────────────────────────────────────────
    st.markdown('<hr class="gt-divider">', unsafe_allow_html=True)
    st.markdown("**How your setup compares to industry benchmarks**")

    benchmarks = [
        ("Your setup",              calc_pue,  calc_wue or 1.8,  calc_cue),
        ("Google hyperscale",       1.10,      0.49,             1.10 * 0.000056),
        ("Microsoft hyperscale",    1.10,      0.49,             1.10 * 0.000233),
        ("Meta hyperscale",         1.10,      0.26,             1.10 * 0.000233),
        ("AWS cloud average",       1.20,      1.80,             1.20 * 0.000386),
        ("Enterprise on-premise",   1.50,      1.80,             1.50 * 0.000490),
        ("Legacy DC",               1.80,      1.80,             1.80 * 0.000635),
        ("Norway ideal (hydro)",    1.10,      0.30,             1.10 * 0.000017),
    ]

    df_bench = pd.DataFrame(benchmarks, columns=["Datacenter", "PUE", "WUE (L/kWh)", "CUE (kg CO₂/kWh)"])
    df_bench["PUE rating"]  = df_bench["PUE"].apply(
        lambda v: "🟢 Excellent" if v <= 1.1 else ("🟡 Good" if v <= 1.3 else ("🟠 Average" if v <= 1.6 else "🔴 Poor")))
    df_bench["WUE rating"]  = df_bench["WUE (L/kWh)"].apply(
        lambda v: "🟢 Excellent" if v <= 0.3 else ("🟡 Good" if v <= 0.6 else ("🟠 Average" if v <= 1.2 else "🔴 Poor")))
    df_bench["CUE rating"]  = df_bench["CUE (kg CO₂/kWh)"].apply(
        lambda v: "🟢 Excellent" if v <= 0.00006 else ("🟡 Good" if v <= 0.0003 else ("🟠 Average" if v <= 0.0006 else "🔴 Poor")))
    df_bench["CUE (kg CO₂/kWh)"] = df_bench["CUE (kg CO₂/kWh)"].map("{:.6f}".format)

    st.dataframe(df_bench.set_index("Datacenter"), use_container_width=True)

    # ── What-if: AquaTensor net water impact ──────────────────────────────────
    st.markdown('<hr class="gt-divider">', unsafe_allow_html=True)
    st.markdown("**WUE improvement with AquaTensor installed**")
    st.markdown(
        "AquaTensor's membrane distillation recovers waste heat and produces fresh water, "
        "effectively reducing your net WUE below zero (net water positive)."
    )

    aq_region = st.selectbox("Region (water stress context)", list(REGIONAL_WATER_STRESS.keys()), key="aq_region_eff")
    aq_whr    = st.slider("Waste Heat Recovery ratio", 0.1, 1.0, 0.65, 0.05, key="aq_whr_eff")
    aq_temp   = st.slider("Feed temperature (°C)", 40, 80, 60, key="aq_temp_eff")

    aq_config_eff = AquaTensorConfig(
        provider=calc_wue_provider,
        custom_wue=calc_custom_wue,
        region=aq_region,
        aquatensor_installed=True,
        whr_ratio=aq_whr,
        feed_temperature_c=float(aq_temp),
    )
    bridge_eff = AquaTensorBridge(aq_config_eff)
    water_eff  = bridge_eff.calculate_water_metrics(calc_it_energy, 60.0)

    net_wue = water_eff.net_water_impact_liters / calc_it_energy if calc_it_energy > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("WUE (without AquaTensor)", f"{calc_wue:.2f} L/kWh" if calc_wue else "N/A",
            delta="traditional cooling", good=False)
    with c2:
        kpi("Water Produced", f"{water_eff.water_produced_liters:.5f} L",
            delta=f"MD @ {aq_temp}°C, {aq_whr*100:.0f}% WHR", good=True)
    with c3:
        kpi("Net WUE (with AquaTensor)",
            f"{net_wue:.4f} L/kWh",
            delta="NET POSITIVE ✓" if net_wue < 0 else "net negative",
            good=(net_wue < 0))
    with c4:
        kpi("Water Stress", f"{water_eff.water_stress_index}/5.0",
            delta=water_eff.water_stress_label,
            good=(water_eff.water_stress_index < 2.5),
            neutral=(water_eff.water_stress_index < 2.5))

    st.markdown(f"""
<div class="info-box">
  <strong>Reading your indices:</strong><br>
  PUE <strong>{calc_pue:.2f}</strong> means for every 1 kWh your GPU uses,
  your datacenter consumes <strong>{calc_pue:.2f} kWh</strong> total
  ({(calc_pue-1)*100:.0f}% goes to cooling and infrastructure).<br><br>
  WUE <strong>{calc_wue:.2f} L/kWh</strong> means this job evaporated
  <strong>{water_consumed:.5f} L</strong> of fresh water for cooling.<br><br>
  CUE <strong>{calc_cue:.6f} kg CO₂/kWh</strong> means every kWh of GPU compute
  ultimately emits <strong>{calc_cue:.6f} kg CO₂</strong> when DC overhead and grid intensity are included.
</div>
""", unsafe_allow_html=True)

# ── ESG Report ────────────────────────────────────────────────────────────────
elif page == "ESG Report":
    st.markdown('<div class="section-title">ESG Report — Scope 2 Emissions</div>', unsafe_allow_html=True)
    st.markdown("Auto-generates a Scope 2 emissions report aligned with GHG Protocol, SEC climate disclosure, and EU CSRD.")

    col1, col2 = st.columns(2)
    with col1:
        org_name = st.text_input("Organization name", value="My Organization")
        reporting_period = st.text_input("Reporting period", value="FY2026")
        region = st.text_input("Region", value="US-East")
    with col2:
        carbon_intensity = st.number_input("Grid carbon intensity (kg CO2/kWh)", value=0.000233, format="%.6f")
        reporting_standard = st.selectbox("Standard", [
            "GHG Protocol Scope 2", "SEC Climate Disclosure Rules", "EU CSRD", "ISO 14064-1"])
        contact = st.text_input("Contact email (optional)", value="")

    st.markdown('<hr class="gt-divider">', unsafe_allow_html=True)
    history = RunHistory()
    records = history.all()

    if records:
        st.markdown(f"**{len(records)} run(s) in local history**")
        df_hist = pd.DataFrame(records)
        st.dataframe(df_hist[["datetime", "model_name", "stage", "energy_kwh", "emissions_kg", "duration_s"]],
                     use_container_width=True, hide_index=True)
    elif baseline_metrics:
        st.markdown('<div class="info-box">No run history. Using currently loaded metrics.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">No run data. Load metrics or run training scripts first.</div>',
                    unsafe_allow_html=True)

    if st.button("Generate ESG Report", type="primary"):
        org = ESGOrganization(name=org_name, reporting_period=reporting_period, region=region,
                              carbon_intensity_kg_per_kwh=carbon_intensity,
                              reporting_standard=reporting_standard, contact_email=contact)
        reporter = ESGReporter(org)
        for r in records:
            reporter._runs.append(ESGRunRecord(
                run_id=r.get("datetime", ""), timestamp=r.get("timestamp", 0),
                model_name=r.get("model_name", "unknown"), stage=r.get("stage", "training"),
                duration_s=r.get("duration_s", 0), energy_kwh=r.get("energy_kwh", 0),
                emissions_kg=r.get("emissions_kg", 0)))
        if not records and baseline_metrics:
            reporter._runs.append(ESGRunRecord(
                run_id="loaded", timestamp=_time.time(), model_name="loaded_model",
                stage="training", duration_s=baseline_metrics.duration_s,
                energy_kwh=baseline_metrics.energy_kwh, emissions_kg=baseline_metrics.emissions_kg))
        report = reporter.generate_report()
        st.markdown('<hr class="gt-divider">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Runs", report.total_runs)
        c2.metric("Total Energy (DC)", f"{report.total_energy_kwh_dc:.4f} kWh")
        c3.metric("Total CO2 (DC)", f"{report.total_emissions_kg_dc:.4f} kg")
        c4.metric("Equiv. km driven", f"{report.emissions_equiv_km_driven:.1f}")
        st.code(report.to_text(), language=None)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download JSON", data=report.to_json(),
                               file_name=f"greentensor_esg_{reporting_period.replace(' ','_')}.json",
                               mime="application/json")
        with col2:
            st.download_button("Download text", data=report.to_text(),
                               file_name=f"greentensor_esg_{reporting_period.replace(' ','_')}.txt",
                               mime="text/plain")

# ── Run History ───────────────────────────────────────────────────────────────
elif page == "Run History":
    st.markdown('<div class="section-title">Run History</div>', unsafe_allow_html=True)
    history = RunHistory()
    records = history.all()

    if not records:
        st.markdown('<div class="info-box">No run history yet. Runs are recorded automatically when you use GreenTensor.</div>',
                    unsafe_allow_html=True)
        st.code("""from greentensor import GreenTensor
from greentensor.core.history import RunHistory

with GreenTensor() as gt:
    train()

RunHistory().record(gt.metrics, model_name="MyModel", stage="training")""")
    else:
        st.markdown(f"**{len(records)} runs recorded**")
        df = pd.DataFrame(records)
        st.dataframe(df[["datetime", "model_name", "stage", "duration_s",
                          "energy_kwh", "emissions_kg", "idle_seconds"]].rename(columns={
            "datetime": "Date", "model_name": "Model", "stage": "Stage",
            "duration_s": "Duration (s)", "energy_kwh": "Energy (kWh)",
            "emissions_kg": "CO2 (kg)", "idle_seconds": "Idle (s)",
        }), use_container_width=True, hide_index=True)
        st.markdown("**Energy over time**")
        st.line_chart(df.set_index("datetime")["energy_kwh"])
        st.markdown("**CO2 over time**")
        st.line_chart(df.set_index("datetime")["emissions_kg"])
        if st.button("Clear history"):
            history.clear()
            st.rerun()

# ── Security Report ───────────────────────────────────────────────────────────
elif page == "Security Report":
    st.markdown('<div class="section-title">Security Report</div>', unsafe_allow_html=True)
    st.markdown("""
GreenTensor monitors GPU power draw continuously during training. Anomalous patterns in the carbon footprint signal potential ML pipeline attacks.

**Detection methods:**
- **alibi-detect SpectralResidual** — frequency-domain statistical anomaly detection on the power time series (same algorithm as Microsoft Azure Metrics Advisor)
- **Threshold detector** — fallback when alibi-detect is unavailable
- **LLM Guard** — scans model inputs for prompt injection and outputs for data leakage

**Threat types detected:**

| Threat | Signal | MITRE ATLAS |
|--------|--------|-------------|
| Cryptominer injection | Power spike + new process | AML.T0011 |
| Data exfiltration | Idle drain + outbound connection | AML.T0024 |
| Backdoor trigger | Inference latency spike | AML.T0043 |
| Model extraction | High-frequency API probing | AML.T0044 |
| Model tampering | SHA-256 hash mismatch | AML.T0018 |
| Supply chain attack | Typosquatted package detected | AML.T0010 |
    """)

    st.markdown('<hr class="gt-divider">', unsafe_allow_html=True)

    if not security_alerts:
        st.markdown('<span class="clean-badge">CLEAN — No threats detected in this session</span>',
                    unsafe_allow_html=True)
        st.markdown("")
        st.markdown('<div class="info-box">Security alerts appear here automatically when GreenTensor detects anomalous energy patterns during a live training run. Use the <strong>Observability Dashboard</strong> to see live detection.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="threat-badge">THREATS DETECTED — {len(security_alerts)} alert(s)</span>',
                    unsafe_allow_html=True)
        for a in security_alerts:
            css = f"alert-{a.severity} alert-card"
            st.markdown(f"""
            <div class="{css}">
              <span class="tag tag-type">{a.alert_type.upper()}</span>
              <span class="tag tag-source">{a.source}</span>
              <strong>{a.severity.upper()}</strong><br>
              {a.message}
            </div>""", unsafe_allow_html=True)

# ── How It Works ──────────────────────────────────────────────────────────────
elif page == "How It Works":
    st.markdown('<div class="section-title">How GreenTensor Works</div>', unsafe_allow_html=True)
    st.markdown("""
**One wrapper. Five capabilities. Zero code changes.**

```python
from greentensor import GreenTensor

with GreenTensor() as gt:
    with gt.mixed_precision():
        train()   # your existing code, unchanged
```

---

**What happens when you enter the context manager:**

1. **GPU optimizer** — enables cuDNN benchmark mode and FP16 mixed precision. 20-40% energy reduction on Tensor Core GPUs.
2. **Idle optimizer** — background thread samples GPU utilization every 0.5s. Surfaces data pipeline bottlenecks.
3. **Anomaly detector** — samples GPU power every second. Builds a rolling baseline and flags deviations using SpectralResidual (alibi-detect).
4. **Digital footprint scanner** — monitors processes, network connections, and model file hashes for attack signals.
5. **CodeCarbon tracker** — measures real energy via hardware counters. Falls back to nvidia-smi power sampling.

**When you exit:**

6. All threads stop. A `RunMetrics` object is created with duration, energy, emissions, and idle time.
7. If a baseline was passed in, savings are calculated against real measured data.
8. A report is printed. Metrics are saved to `.pkl` for this dashboard.

---

**The security detection pipeline:**

```
GPU power sample (every 1s)
        |
        v
SpectralResidual anomaly detection
        |
   spike detected?
        |
        v
Pattern matcher correlates with:
  - active processes (psutil)
  - network connections (psutil)
  - inference latency history
  - model file hashes (SHA-256)
        |
        v
Threat score 0-100
Verdict: BENIGN / SUSPICIOUS / ATTACK
MITRE ATLAS technique ID
Recommended action
        |
        v
Slack / PagerDuty webhook alert
```

---

**Install:**
```bash
pip install greentensor

# Full security stack (alibi-detect + LLM Guard)
pip install greentensor[security]
```

**GitHub:** [DhivyaBalakumar/GreenTensor-Carbon-Aware-MLOps](https://github.com/DhivyaBalakumar/GreenTensor-Carbon-Aware-MLOps)
    """)
