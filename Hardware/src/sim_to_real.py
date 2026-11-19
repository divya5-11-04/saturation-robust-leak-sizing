"""
Sim-to-Real Cross-Domain Validation and Physics Alignment.
Analyzes physics consistency between FluidSIM simulation models and experimental hardware.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_sim_to_real_analysis():
    hw_dir = Path(__file__).resolve().parents[1]
    sim_dir = hw_dir.parent / "Simulation"
    sys.path.append(str(hw_dir / "src"))
    from hardware_features import load_hardware_dataset
    
    sim_train_p = sim_dir / "data" / "processed" / "train_tabular.csv"
    sim_test_p = sim_dir / "data" / "processed" / "test_tabular.csv"
    
    if not sim_train_p.exists():
        print("Simulation feature files missing for sim-to-real.")
        return
        
    df_sim = pd.concat([pd.read_csv(sim_train_p), pd.read_csv(sim_test_p)], ignore_index=True)
    df_hw, _ = load_hardware_dataset(hw_dir / "hardware_dataset")
    df_hw_runs = df_hw[df_hw["segment_idx"] == 0]
    
    # Shared leak levels
    shared_leaks = [0.0, 25.0, 50.0, 75.0, 100.0]
    
    # Aggregate simulation features across shared levels
    sim_agg = df_sim[df_sim["leak_pct"].isin(shared_leaks)].groupby("leak_pct").agg({
        "pressure_bar_mean": "mean",
        "pressure_bar_min": "mean",
        "flow_rate_lpm_mean": "mean",
        "pneumatic_power_W_mean": "mean",
        "mean_leak_conductance": "mean"
    }).reset_index()
    
    # Aggregate hardware features across shared levels
    hw_agg = df_hw_runs[df_hw_runs["leak_pct"].isin(shared_leaks)].groupby("leak_pct").agg({
        "p_mean": "mean",
        "p_min": "mean",
        "q_mean": "mean",
        "power_mean": "mean",
        "conductance_proxy": "mean",
        "stroke_period": "mean"
    }).reset_index()
    
    # Pearson cross-domain correlations
    p_corr = float(np.corrcoef(sim_agg["pressure_bar_mean"], hw_agg["p_mean"])[0, 1])
    power_corr = float(np.corrcoef(sim_agg["pneumatic_power_W_mean"], hw_agg["power_mean"])[0, 1])
    cond_corr = float(np.corrcoef(sim_agg["mean_leak_conductance"], hw_agg["conductance_proxy"])[0, 1])
    
    sim_to_real_metrics = {
        "shared_leak_levels": shared_leaks,
        "pressure_correlation": round(p_corr, 4),
        "power_correlation": round(power_corr, 4),
        "conductance_correlation": round(cond_corr, 4),
        "sim_p_mean": sim_agg["pressure_bar_mean"].round(3).tolist(),
        "hw_p_mean": hw_agg["p_mean"].round(3).tolist(),
        "sim_power_mean": sim_agg["pneumatic_power_W_mean"].round(3).tolist(),
        "hw_power_mean": hw_agg["power_mean"].round(3).tolist(),
        "hw_stroke_period": hw_agg["stroke_period"].round(3).tolist()
    }
    
    out_json = hw_dir / "results" / "metrics" / "sim_to_real_alignment.json"
    out_json.write_text(json.dumps(sim_to_real_metrics, indent=2))
    
    # Also save as readable CSV table
    comparison_df = pd.DataFrame({
        "Leak (%)": shared_leaks,
        "Sim Line Pressure (bar)": sim_agg["pressure_bar_mean"].round(2),
        "HW Line Pressure (bar)": hw_agg["p_mean"].round(2),
        "Sim Power (W)": sim_agg["pneumatic_power_W_mean"].round(1),
        "HW Power (W)": hw_agg["power_mean"].round(1),
        "HW Cycle Period (s)": hw_agg["stroke_period"].round(2)
    })
    out_csv = hw_dir / "results" / "metrics" / "sim_to_real_comparison.csv"
    comparison_df.to_csv(out_csv, index=False)
    
    print("\n" + "="*70)
    print("SIM-TO-REAL CROSS-DOMAIN ALIGNMENT SUMMARY")
    print("="*70)
    print(comparison_df.to_string(index=False))
    print(f"\nPressure Sag Correlation: {p_corr:.4f}")
    print(f"Fluid Power Dissipation Correlation: {power_corr:.4f}")
    print(f"Orifice Conductance Correlation: {cond_corr:.4f}")
    print(f"Saved metrics to: {out_json}")
    print(f"Saved comparison table to: {out_csv}")

if __name__ == "__main__":
    run_sim_to_real_analysis()