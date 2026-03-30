import streamlit as st
import pickle
import os
from greentensor.report.metrics import RunMetrics, calculate_savings

st.set_page_config(page_title="GreenTensor Dashboard", page_icon="🌿")
st.title("🌿 GreenTensor Impact Dashboard")

st.sidebar.header("Data Source")
mode = st.sidebar.radio("Input mode", ["Manual", "Load from file"])

baseline_metrics = None
optimized_metrics = None

if mode == "Manual":
    st.subheader("Manual Input")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline**")
        b_energy = st.number_input("Energy (kWh)", key="b_e", min_value=0.0, format="%.6f")
        b_emissions = st.number_input("Emissions (kg CO₂)", key="b_em", min_value=0.0, format="%.6f")
        b_duration = st.number_input("Duration (s)", key="b_d", min_value=0.0)
    with col2:
        st.markdown("**Optimized**")
        o_energy = st.number_input("Energy (kWh)", key="o_e", min_value=0.0, format="%.6f")
        o_emissions = st.number_input("Emissions (kg CO₂)", key="o_em", min_value=0.0, format="%.6f")
        o_duration = st.number_input("Duration (s)", key="o_d", min_value=0.0)

    if b_energy > 0 and o_energy > 0:
        baseline_metrics = RunMetrics(duration_s=b_duration, energy_kwh=b_energy, emissions_kg=b_emissions)
        optimized_metrics = RunMetrics(duration_s=o_duration, energy_kwh=o_energy, emissions_kg=o_emissions)

else:
    b_file = st.file_uploader("Upload baseline_metrics.pkl", type="pkl", key="b_pkl")
    o_file = st.file_uploader("Upload optimized_metrics.pkl", type="pkl", key="o_pkl")

    if b_file:
        baseline_metrics = pickle.load(b_file)
        st.success("Baseline loaded.")
    if o_file:
        optimized_metrics = pickle.load(o_file)
        st.success("Optimized loaded.")

    # Also auto-load from disk if running locally
    if not baseline_metrics and os.path.exists("baseline_metrics.pkl"):
        with open("baseline_metrics.pkl", "rb") as f:
            baseline_metrics = pickle.load(f)
        st.info("Auto-loaded baseline_metrics.pkl from disk.")

# ── Results ──────────────────────────────────────────────────────────────────

if baseline_metrics and optimized_metrics:
    savings = calculate_savings(baseline_metrics, optimized_metrics)

    st.subheader("Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Energy Reduction", f"{savings['energy_reduction_pct']:.1f}%")
    c2.metric("Energy Saved", f"{savings['energy_saved_kwh']:.6f} kWh")
    c3.metric("CO₂ Saved", f"{savings['emissions_saved_kg']:.6f} kg")
    c4.metric("Time Saved", f"{savings['time_saved_s']:.2f} s")

    st.subheader("Energy Comparison")
    st.bar_chart({
        "Baseline": baseline_metrics.energy_kwh,
        "Optimized": optimized_metrics.energy_kwh,
    })

elif baseline_metrics:
    st.subheader("Baseline Run")
    st.metric("Energy", f"{baseline_metrics.energy_kwh:.6f} kWh")
    st.metric("Emissions", f"{baseline_metrics.emissions_kg:.6f} kg CO₂")
    st.metric("Duration", f"{baseline_metrics.duration_s:.2f} s")
    st.info("Upload or enter optimized metrics to see savings.")
