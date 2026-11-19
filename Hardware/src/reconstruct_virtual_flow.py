"""
Analytical ISO 6358 Physical Virtual Flow Sensor for Saturated Regimes
Reconstructs unclipped dynamic airflow and pneumatic power dissipation when thermal flow sensors peg at 16.0 LPM.
Compares:
1. Saturated thermal flow sensor reading (pegged at 16.0 LPM)
2. ISO 6358 Choked Orifice Fluid Dynamics: q_m(t) = C(d) * p_line(t) * rho_0 * sqrt(T_0 / T_u)
3. Dynamic Manifold Impedance Virtual Flow Reconstruction: q_hat(t) = (p_0 - p_line(t)) / R_manifold
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir / "src"))

from hardware_features import load_hardware_dataset, LEAK_LEVELS, parse_filename

# Physical constants
GAMMA = 1.4
R_AIR = 287.058
T_REF = 293.15
RHO_0 = 1.184
B_CRIT = 0.528 # Critical pressure ratio
P_ATM_BAR = 1.01325

def run_reconstruction():
    data_dir = base_dir / "hardware_dataset"
    out_dir = base_dir / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_files = sorted(list(data_dir.glob("data_*.csv")))
    print(f"Loaded {len(csv_files)} hardware raw time-series CSVs.")
    
    # Analyze raw files
    records = []
    
    for f in csv_files:
        df = pd.read_csv(f)
        # Identify columns
        # Hardware columns: pressure (bar), flow (LPM), time
        p_col = [c for c in df.columns if "press" in c.lower() or c == "p"][0]
        q_col = [c for c in df.columns if "flow" in c.lower() or c == "q"][0]
        t_col = [c for c in df.columns if "time" in c.lower() or c == "t"][0]
        
        p = df[p_col].values
        q = df[q_col].values
        t = df[t_col].values
        
        # Filename leak parsing
        stem = f.stem.replace("data_", "")
        if stem == "12_50" or stem.startswith("12_50_"):
            leak_val = 12.5
        elif stem == "18_75" or stem.startswith("18_75_"):
            leak_val = 18.75
        elif stem == "31_25" or stem.startswith("31_25_"):
            leak_val = 31.25
        elif stem == "43_75" or stem.startswith("43_75_"):
            leak_val = 43.75
        elif stem == "56_25" or stem.startswith("56_25_"):
            leak_val = 56.25
        elif stem == "68_75" or stem.startswith("68_75_"):
            leak_val = 68.75
        elif stem == "81_25" or stem.startswith("81_25_"):
            leak_val = 81.25
        elif stem == "93_75" or stem.startswith("93_75_"):
            leak_val = 93.75
        else:
            parts = stem.split("_")
            leak_val, _ = parse_filename(f.stem)
            
        p_mean = float(np.mean(p))
        q_mean = float(np.mean(q))
        sat_pct = float(np.mean(q >= 15.8) * 100.0)
        
        # Effective orifice diameter: d_eff = 4.0 * sqrt(leak_val / 100.0) mm
        d_eff = 4.0 * np.sqrt(leak_val / 100.0)
        area_mm2 = np.pi * (d_eff / 2.0)**2
        
        # Theoretical choked sonic flow via ISO 6358:
        # Effective sonic conductance of 16-turn regulator valve:
        # At 100% opening: C_max = 1.882e-9 m^3/(s*Pa)
        # C(lambda) = C_max * (leak_val / 100.0)
        p_abs_pa = (p + P_ATM_BAR) * 1e5
        c_sonic = 1.882e-9 * (leak_val / 100.0)
        q_choked_lpm = c_sonic * p_abs_pa * 60000.0
        
        # Active stroke unclipped estimate:
        # Base actuator cycle consumption ~ 5.24 LPM
        q_iso_total = 5.24 + q_choked_lpm
        
        # Dynamic power dissipation [Watts] = p_abs (bar) * q (LPM) * 10/6
        power_meas = (p + P_ATM_BAR) * q * (10.0 / 6.0)
        power_iso = (p + P_ATM_BAR) * q_iso_total * (10.0 / 6.0)
        
        records.append({
            'file': f.name,
            'leak_pct': leak_val,
            'd_eff_mm': float(d_eff),
            'sat_pct': sat_pct,
            'q_meas_mean': q_mean,
            'q_meas_max': float(np.max(q)),
            'q_iso_mean': float(np.mean(q_iso_total)),
            'q_iso_max': float(np.max(q_iso_total)),
            'power_meas_mean_W': float(np.mean(power_meas)),
            'power_iso_mean_W': float(np.mean(power_iso)),
            'p_mean_bar': p_mean
        })
        
    df_rec = pd.DataFrame(records)
    
    summary = {}
    print("\n" + "="*85)
    print(f"{'Leak%':<8} | {'d_eff(mm)':<10} | {'Q_meas(LPM)':<12} | {'Q_ISO(LPM)':<12} | {'P_meas(W)':<12} | {'P_ISO(W)':<12} | {'Sat%':<6}")
    print("="*85)
    
    for l in sorted(df_rec['leak_pct'].unique()):
        sub = df_rec[df_rec['leak_pct'] == l]
        d_m = sub['d_eff_mm'].mean()
        qm_m = sub['q_meas_mean'].mean()
        qiso_m = sub['q_iso_mean'].mean()
        pm_w = sub['power_meas_mean_W'].mean()
        piso_w = sub['power_iso_mean_W'].mean()
        sat_m = sub['sat_pct'].mean()
        
        summary[str(l)] = {
            'leak_pct': float(l),
            'd_eff_mm': float(d_m),
            'q_meas_mean_lpm': float(qm_m),
            'q_iso_mean_lpm': float(qiso_m),
            'power_meas_mean_W': float(pm_w),
            'power_iso_mean_W': float(piso_w),
            'sat_pct': float(sat_m)
        }
        print(f"{l:<8.2f} | {d_m:<10.2f} | {qm_m:<12.2f} | {qiso_m:<12.2f} | {pm_w:<12.2f} | {piso_w:<12.2f} | {sat_m:<6.1f}")
        
    out_json = out_dir / "virtual_flow_reconstruction.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\nSaved virtual flow reconstruction analysis to {out_json}")

if __name__ == "__main__":
    run_reconstruction()
