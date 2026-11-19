"""
High-Precision Physics-Consistent Digitization of FluidSIM Simulation Traces
Extracts:
1. Cylinder displacement x(t) [mm]
2. Cylinder velocity v(t) [m/s]
3. Mass flow rate q(t) [l/min]
4. Accumulated volume V_total(t) [l]
5. Line pressure p(t) [bar]
Resampled to 100 Hz (dt = 0.01 s) from ~75 samples/s of native plot resolution.
"""

import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(SIM_DIR, "data", "raw_extracted")
FIG_DIR = os.path.join(SIM_DIR, "results", "figures")
DIAG_DIR = os.path.join(SIM_DIR, "results", "digitization_overlays")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)

# Exact calibrated bounds for each image matching labelled ticks and correct pressure spans:
# (y_top, y_bottom, value_top, value_bottom)
DATASETS_CFG = {
    '0pct': {
        'file': '0% Leak.png',
        'leak_pct': 0.0,
        't_cols': (829, 1618),
        'y_x': (295, 356, 100.0, 0.0),
        'y_v': (374, 419, 0.0, -0.6),
        'y_q': (469, 517, 0.0, -590.0),
        'y_V': (538, 605, 0.0, -1.7),
        'y_p': (651, 816, 6.0, 5.6),
        'y_p_limit': 820
    },
    '1pct': {
        'file': '1% Leak.png',
        'leak_pct': 1.0,
        't_cols': (829, 1578),
        'y_x': (293, 354, 100.0, 0.0),
        'y_v': (370, 416, 0.0, -0.6),
        'y_q': (463, 510, 0.0, -557.0),
        'y_V': (530, 598, 0.0, -1.9),
        'y_p': (635, 800, 6.0, 5.6),
        'y_p_limit': 804
    },
    '5pct': {
        'file': '5% Leak.png',
        'leak_pct': 5.0,
        't_cols': (826, 1578),
        'y_x': (286, 348, 100.0, 0.0),
        'y_v': (365, 412, 0.0, -0.6),
        'y_q': (456, 505, 0.0, -553.0),
        'y_V': (525, 595, 0.0, -3.0),
        'y_p': (677, 803, 5.9, 5.6),
        'y_p_limit': 807
    },
    '25pct': {
        'file': '25% Leak.png',
        'leak_pct': 25.0,
        't_cols': (827, 1575),
        'y_x': (238, 300, 100.0, 0.0),
        'y_v': (317, 363, 0.0, -0.6),
        'y_q': (406, 453, 0.0, -563.0),
        'y_V': (477, 545, 0.0, -8.0),
        'y_p': (622, 845, 5.9, 5.6),
        'y_p_limit': 849
    },
    '50pct': {
        'file': '50%.png',
        'leak_pct': 50.0,
        't_cols': (826, 1578),
        'y_x': (303, 365, 100.0, 0.0),
        'y_v': (382, 428, 0.0, -0.6),
        'y_q': (471, 535, 0.0, -743.0),
        'y_V': (543, 611, 0.0, -15.0),
        'y_p': (654, 770, 5.9, 5.6),
        'y_p_limit': 774
    },
    '75pct': {
        'file': '75 %.png',
        'leak_pct': 75.0,
        't_cols': (825, 1577),
        'y_x': (303, 365, 100.0, 0.0),
        'y_v': (381, 427, 0.0, -0.6),
        'y_q': (469, 532, 0.0, -789.0),
        'y_V': (545, 611, 0.0, -20.0),
        'y_p': (684, 795, 5.8, 5.5),
        'y_p_limit': 799
    },
    '100pct': {
        'file': '100% Leak.png',
        'leak_pct': 100.0,
        't_cols': (825, 1578),
        'y_x': (303, 365, 100.0, 0.0),
        'y_v': (382, 428, 0.0, -0.6),
        'y_q': (471, 534, 0.0, -789.0),
        'y_V': (543, 611, 0.0, -25.0),
        'y_p': (664, 780, 5.8, 5.5),
        'y_p_limit': 784
    }
}

def extract_curve_blue(b, r, y_min, y_max, cols, reject_grid=False):
    """
    Blue curve extraction (x, q, p).
    Captures downward vertical spike tips using idx.max() when span > 6.
    Rejects gridlines (which have r > 120) when reject_grid is True.
    """
    vals = []
    for c in cols:
        sub_b = b[y_min:y_max, c].astype(int)
        sub_r = r[y_min:y_max, c].astype(int)
        diff = sub_b - sub_r
        
        if reject_grid:
            # Curve is saturated dark blue (sub_r < 110), whereas gridlines have sub_r > 125
            idx = np.where((diff > 45) & (sub_b > 115) & (sub_r < 110))[0]
            if len(idx) == 0:
                max_d = np.max(diff)
                if max_d > 60:
                    idx = np.where(diff >= max_d - 15)[0]
        else:
            idx = np.where((diff > 45) & (sub_b > 115))[0]
            
        if len(idx) > 0:
            span = idx.max() - idx.min()
            vals.append(y_min + (idx.max() if span > 6 else np.median(idx)))
        else:
            vals.append(np.nan)
    return np.array(vals)

def extract_curve_steel(b, r, gray, y_min, y_max, cols):
    """
    Steel-slate curve extraction (v, V_total).
    Captures downward vertical spike tips using idx.max() when span > 6.
    """
    vals = []
    for c in cols:
        sub_b = b[y_min:y_max, c].astype(int)
        sub_r = r[y_min:y_max, c].astype(int)
        sub_g = gray[y_min:y_max, c]
        diff = sub_b - sub_r
        idx = np.where(((diff >= 6) & (sub_r < 215)) | ((sub_g < 170) & (diff >= 3)))[0]
        if len(idx) > 0:
            span = idx.max() - idx.min()
            vals.append(y_min + (idx.max() if span > 6 else np.median(idx)))
        else:
            vals.append(np.nan)
    return np.array(vals)

def fill_nan(t, s):
    valid = ~np.isnan(s)
    if np.sum(valid) < 2:
        return np.zeros_like(t)
    f = interp1d(t[valid], s[valid], kind='linear', fill_value='extrapolate')
    return f(t)

def process_all():
    t_eval = np.linspace(0.0, 10.0, 1001)
    results = {}
    
    for tag, cfg in DATASETS_CFG.items():
        img_path = os.path.join(SIM_DIR, cfg['file'])
        img = cv2.imread(img_path)
        overlay_img = img.copy()
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        c_start, c_end = cfg['t_cols']
        cols = np.arange(c_start, c_end + 1)
        t_raw = np.linspace(0.0, 10.0, len(cols))
        
        # 1. Displacement x
        y1, y2, v1, v2 = cfg['y_x']
        y_pts = extract_curve_blue(b, r, y1 - 4, y2 + 4, cols)
        y_filled = fill_nan(t_raw, y_pts)
        x_raw = v1 + (y_filled - y1) / (y2 - y1) * (v2 - v1)
        x_eval = interp1d(t_raw, x_raw, fill_value='extrapolate')(t_eval)
        x_eval = np.clip(x_eval, 0.0, 100.0)
        
        # 2. Velocity v
        y1, y2, v1, v2 = cfg['y_v']
        y_pts_v = extract_curve_steel(b, r, gray, y1 - 4, y2 + 6, cols)
        y_filled_v = fill_nan(t_raw, y_pts_v)
        v_raw = v1 + (y_filled_v - y1) / (y2 - y1) * (v2 - v1)
        v_eval = interp1d(t_raw, v_raw, fill_value='extrapolate')(t_eval)
        v_eval = np.clip(v_eval, -0.8, 0.4)
        
        # 3. Flow rate q
        y1, y2, v1, v2 = cfg['y_q']
        y_pts_q = extract_curve_blue(b, r, y1 - 4, y2 + 10, cols)
        y_filled_q = fill_nan(t_raw, y_pts_q)
        q_raw = v1 + (y_filled_q - y1) / (y2 - y1) * (v2 - v1)
        q_eval = interp1d(t_raw, q_raw, fill_value='extrapolate')(t_eval)
        q_eval = np.minimum(q_eval, 0.0)
        
        # 4. Total Volume V
        y1, y2, v1, v2 = cfg['y_V']
        y_pts_V = extract_curve_steel(b, r, gray, y1 - 4, y2 + 6, cols)
        y_filled_V = fill_nan(t_raw, y_pts_V)
        V_raw = v1 + (y_filled_V - y1) / (y2 - y1) * (v2 - v1)
        V_eval = interp1d(t_raw, V_raw, fill_value='extrapolate')(t_eval)
        V_eval = np.minimum(V_eval, 0.0)
        V_eval = np.minimum.accumulate(V_eval)
        
        # 5. Pressure p
        y1, y2, v1, v2 = cfg['y_p']
        y_search_top = min(630, min(y1, y2) - 30)
        y_search_bot = cfg.get('y_p_limit', max(y1, y2) + 10)
        y_pts_p = extract_curve_blue(b, r, y_search_top, y_search_bot, cols, reject_grid=True)
        y_filled_p = fill_nan(t_raw, y_pts_p)
        p_raw = v1 + (y_filled_p - y1) / (y2 - y1) * (v2 - v1)
        p_eval = interp1d(t_raw, p_raw, fill_value='extrapolate')(t_eval)
        p_eval = np.clip(p_eval, 5.2, 6.05)
        
        # Diagnostic overlay
        for idx_c, col in enumerate(cols):
            if not np.isnan(y_pts_p[idx_c]):
                overlay_img[int(y_pts_p[idx_c]), col] = [0, 255, 255] # Yellow for pressure
            if not np.isnan(y_pts_q[idx_c]):
                overlay_img[int(y_pts_q[idx_c]), col] = [255, 0, 255] # Magenta for flow
        
        diag_path = os.path.join(DIAG_DIR, f"{tag}_overlay_check.png")
        cv2.imwrite(diag_path, overlay_img[600:850, c_start:c_end])
        
        df = pd.DataFrame({
            'time_s': np.round(t_eval, 3),
            'displacement_mm': np.round(x_eval, 3),
            'velocity_mps': np.round(v_eval, 4),
            'flow_rate_lpm': np.round(q_eval, 2),
            'total_volume_l': np.round(V_eval, 4),
            'pressure_bar': np.round(p_eval, 4),
            'leak_pct': cfg['leak_pct']
        })
        
        out_csv = os.path.join(OUT_DIR, f"{tag}_leak.csv")
        df.to_csv(out_csv, index=False)
        results[tag] = df
        print(f"[{tag}] Digitized successfully: x_max={x_eval.max():.1f}mm, v_min={v_eval.min():.2f}m/s, q_min={q_eval.min():.1f}lpm, V_end={V_eval[-1]:.2f}L, p_range=({p_eval.min():.3f}, {p_eval.max():.3f})bar, p_median={np.median(p_eval):.3f}bar")
        
    # Generate multi-signal overview figure (Figure 1 for paper)
    fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    colors = {
        '0pct': '#1f77b4', '1pct': '#2ca02c', '5pct': '#ff7f0e',
        '25pct': '#d62728', '50pct': '#9467bd', '75pct': '#8c564b', '100pct': '#e377c2'
    }
    
    for tag, df in results.items():
        c = colors[tag]
        lbl = f"{df['leak_pct'].iloc[0]:.0f}% Leak"
        axs[0].plot(df['time_s'], df['displacement_mm'], label=lbl, color=c, lw=1.5)
        axs[1].plot(df['time_s'], df['velocity_mps'], label=lbl, color=c, lw=1.2)
        axs[2].plot(df['time_s'], df['flow_rate_lpm'], label=lbl, color=c, lw=1.2)
        axs[3].plot(df['time_s'], df['total_volume_l'], label=lbl, color=c, lw=1.5)
        axs[4].plot(df['time_s'], df['pressure_bar'], label=lbl, color=c, lw=1.5)
        
    axs[0].set_ylabel('Displacement [mm]', fontsize=11, fontweight='bold')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend(loc='upper right', ncol=4, fontsize=9)
    axs[0].set_ylim(-5, 105)
    
    axs[1].set_ylabel('Velocity [m/s]', fontsize=11, fontweight='bold')
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].set_ylim(-0.75, 0.15)
    
    axs[2].set_ylabel('Flow Rate [L/min]', fontsize=11, fontweight='bold')
    axs[2].grid(True, linestyle='--', alpha=0.6)
    axs[2].set_ylim(-850, 50)
    
    axs[3].set_ylabel('Total Volume [L]', fontsize=11, fontweight='bold')
    axs[3].grid(True, linestyle='--', alpha=0.6)
    axs[3].set_ylim(-27, 2)
    
    axs[4].set_ylabel('Pressure [bar]', fontsize=11, fontweight='bold')
    axs[4].set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    axs[4].grid(True, linestyle='--', alpha=0.6)
    axs[4].set_ylim(5.3, 6.05)
    axs[4].set_xlim(0, 10.0)
    
    plt.suptitle('Multi-State Dynamic Trajectories across Pneumatic Leak Levels (0% to 100%)', fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    out_fig = os.path.join(FIG_DIR, "fig1_multisignal_traces.png")
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved digitized multi-signal plot: {out_fig}")

if __name__ == "__main__":
    process_all()
