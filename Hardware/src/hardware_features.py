"""
Hardware Dataset Loader and Feature Extraction for Pneumatic Leak Detection.
Extracts thermodynamic, kinematic, and statistical features across all 105 experimental runs.
"""

from pathlib import Path
import glob
import re
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

LEAK_LEVELS = [0.0, 6.25, 12.5, 18.75, 25.0, 31.25, 37.5, 43.75, 50.0, 56.25, 62.5, 68.75, 75.0, 81.25, 87.5, 93.75, 100.0]
LEAK_TO_CLASS = {val: idx for idx, val in enumerate(LEAK_LEVELS)}
CLASS_TO_LEAK = {idx: val for idx, val in enumerate(LEAK_LEVELS)}

def parse_filename(stem: str) -> tuple[float, int]:
    s = stem.replace("data_", "")
    if s == "12_50":
        return 12.5, 0
    elif s.startswith("12_50_"):
        return 12.5, int(s.split("_")[-1])
    
    for l in sorted(LEAK_LEVELS, reverse=True):
        s_repr = str(l).replace(".", "_")
        if s_repr.endswith("_0"):
            s_repr = s_repr[:-2]
        if s == s_repr:
            return l, 0
        elif s.startswith(s_repr + "_"):
            suffix = s[len(s_repr) + 1:]
            return l, int(suffix)
    raise ValueError(f"Could not parse filename stem: {stem}")

def get_coarse_stage(leak: float) -> str:
    """Industrial 4-stage health classification."""
    if leak == 0.0:
        return "Healthy"
    elif leak <= 18.75:
        return "Minor"
    elif leak <= 50.0:
        return "Moderate"
    else:
        return "Severe"

COARSE_STAGE_MAP = {"Healthy": 0, "Minor": 1, "Moderate": 2, "Severe": 3}

def extract_kinematic_features(p: np.ndarray, t: np.ndarray) -> dict[str, float]:
    """Detects double-acting cylinder strokes from cyclical pressure dips."""
    inv_p = -p
    p_std = max(float(np.std(p)), 0.005)
    peaks, props = find_peaks(inv_p, distance=4, prominence=0.25 * p_std)
    dur = float(t[-1] - t[0]) if len(t) > 1 else 10.0
    num_strokes = len(peaks)
    stroke_freq = float(num_strokes / dur) if dur > 0 else 0.0
    stroke_period = float(1.0 / stroke_freq) if stroke_freq > 0.1 else float(dur)
    
    if len(peaks) > 1:
        peak_times = t[peaks]
        intervals = np.diff(peak_times)
        period_std = float(np.std(intervals))
        period_cv = float(period_std / (np.mean(intervals) + 1e-6))
    else:
        period_std = 0.0
        period_cv = 0.0
        
    return {
        "num_strokes": float(num_strokes),
        "stroke_freq": stroke_freq,
        "stroke_period": stroke_period,
        "period_std": period_std,
        "period_cv": period_cv,
    }

def extract_physics_and_stats(p: np.ndarray, q: np.ndarray, t: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Computes comprehensive thermodynamic, aerodynamic, and kinematic features."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    t = np.asarray(t, dtype=float)
    
    # Statistical Pressure
    p_mean = float(np.mean(p))
    p_std = float(np.std(p))
    p_min = float(np.min(p))
    p_max = float(np.max(p))
    p_range = p_max - p_min
    p_q10 = float(np.quantile(p, 0.10))
    p_q90 = float(np.quantile(p, 0.90))
    p_slope = float(np.polyfit(np.arange(len(p)), p, 1)[0]) if len(p) > 1 else 0.0
    p_energy = float(np.mean(np.square(p)))
    
    # Statistical Airflow
    q_mean = float(np.mean(q))
    q_std = float(np.std(q))
    q_min = float(np.min(q))
    q_max = float(np.max(q))
    q_range = q_max - q_min
    q_q10 = float(np.quantile(q, 0.10))
    q_q90 = float(np.quantile(q, 0.90))
    q_slope = float(np.polyfit(np.arange(len(q)), q, 1)[0]) if len(q) > 1 else 0.0
    q_energy = float(np.mean(np.square(q)))
    
    # Sensor Operational Regimes
    q_sat_ratio = float(np.mean(q >= 15.8)) # pegged at 16 LPM saturation
    q_zero_ratio = float(np.mean(q <= 0.2)) # cylinder dwell
    q_active_ratio = float(np.mean((q > 0.2) & (q < 15.8))) # transition
    
    # Kinematics
    kin = extract_kinematic_features(p, t)
    
    # Thermodynamic Coupling
    p_abs = p + 1.013 # Gauge to Absolute bar
    # Fluid Power: P = p_abs (bar) * 1e5 (Pa) * q (LPM) * (1e-3/60 m^3/s) = p_abs * q * 10/6 Watts
    fluid_power = p_abs * q * (10.0 / 6.0)
    power_mean = float(np.mean(fluid_power))
    power_max = float(np.max(fluid_power))
    power_energy = float(np.mean(np.square(fluid_power)))
    
    # Conductance Proxy: q / sqrt(p_abs - 1.013)
    dp_atm = np.maximum(p, 0.05)
    conductance_proxy = float(np.mean((q / 60.0) / np.maximum(p_abs, 0.05)))
    
    # Stroke Energy Metric: Power * Cycle Period
    stroke_specific_energy = power_mean * kin["stroke_period"]
    
    # Cross correlation
    if p_std > 1e-4 and q_std > 1e-4:
        p_q_corr = float(np.corrcoef(p, q)[0, 1])
    else:
        p_q_corr = 0.0
        
    feat = {
        "p_mean": p_mean,
        "p_std": p_std,
        "p_min": p_min,
        "p_max": p_max,
        "p_range": p_range,
        "p_q10": p_q10,
        "p_q90": p_q90,
        "p_slope": p_slope,
        "p_energy": p_energy,
        "q_mean": q_mean,
        "q_std": q_std,
        "q_min": q_min,
        "q_max": q_max,
        "q_range": q_range,
        "q_q10": q_q10,
        "q_q90": q_q90,
        "q_slope": q_slope,
        "q_energy": q_energy,
        "q_sat_ratio": q_sat_ratio,
        "q_zero_ratio": q_zero_ratio,
        "q_active_ratio": q_active_ratio,
        "power_mean": power_mean,
        "power_max": power_max,
        "power_energy": power_energy,
        "conductance_proxy": conductance_proxy,
        "p_q_corr": p_q_corr,
        "stroke_specific_energy": stroke_specific_energy,
        "p_to_freq_ratio": float(p_mean / (kin["stroke_freq"] + 1e-4)),
    }
    feat.update(kin)
    
    if prefix:
        return {f"{prefix}_{k}": v for k, v in feat.items()}
    return feat

def split_run_into_windows(df: pd.DataFrame, num_segments: int = 4) -> list[pd.DataFrame]:
    """Splits a single run into full-duration and temporal sub-windows."""
    segments = [df] # 0: full run
    n = len(df)
    third = n // 3
    if third >= 20:
        segments.append(df.iloc[:third])
        segments.append(df.iloc[third:2 * third])
        segments.append(df.iloc[2 * third:])
    return segments

def load_hardware_dataset(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Loads all 105 CSV files, computes features for full runs and temporal segments."""
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {data_dir}")
        
    records = []
    for f in files:
        stem = f.stem
        leak, run_idx = parse_filename(stem)
        coarse_stage = get_coarse_stage(leak)
        class_17 = LEAK_TO_CLASS[leak]
        coarse_class = COARSE_STAGE_MAP[coarse_stage]
        # Equivalent Orifice Diameter in mm: d_leak = 4.0 * sqrt(leak / 100)
        d_leak_mm = 4.0 * np.sqrt(leak / 100.0)
        
        df = pd.read_csv(f)
        segments = split_run_into_windows(df, num_segments=4)
        
        for seg_idx, seg in enumerate(segments):
            p = seg["Pressure_bar"].values
            q = seg["Airflow_LPM"].values
            t = seg["Elapsed_Time_sec"].values
            
            feat = extract_physics_and_stats(p, q, t)
            feat["leak_pct"] = leak
            feat["class_17"] = class_17
            feat["coarse_stage"] = coarse_stage
            feat["coarse_class"] = coarse_class
            feat["d_leak_mm"] = d_leak_mm
            feat["run_idx"] = run_idx
            feat["group_id"] = stem # run filename acts as strict group ID
            feat["segment_idx"] = seg_idx
            feat["num_points"] = len(p)
            
            records.append(feat)
            
    df_all = pd.DataFrame(records)
    meta_cols = ["leak_pct", "class_17", "coarse_stage", "coarse_class", "d_leak_mm", "run_idx", "group_id", "segment_idx", "num_points"]
    feature_cols = [c for c in df_all.columns if c not in meta_cols]
    return df_all, feature_cols


