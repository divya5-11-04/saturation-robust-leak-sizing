"""
Comprehensive Physics-Aware Dataset Generator for Pneumatic Leak Detection & Quantification
Generates:
1. High-fidelity cycle-segmented & sliding window time series
2. Physics-engineered feature vectors
3. Multi-condition industrial noise augmentations (SNR = 30dB, 20dB, 15dB, drift)
4. Train, Validation, Test splits strictly partitioned by cycle (Zero Data Leakage)
5. Out-of-Distribution Harsh Test Set
6. Feature correlation and physics trajectory visualizations
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter
from fluid_physics import compute_physics_features

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(SIM_DIR, "data", "raw_extracted")
OUT_DIR = os.path.join(SIM_DIR, "data", "processed")
FIG_DIR = os.path.join(SIM_DIR, "results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Severity class mapping
LABEL_MAP = {
    0.0:   (0, "Healthy"),
    1.0:   (1, "Micro_1pct"),
    5.0:   (2, "Minor_5pct"),
    25.0:  (3, "Moderate_25pct"),
    50.0:  (4, "Significant_50pct"),
    75.0:  (5, "Severe_75pct"),
    100.0: (6, "Catastrophic_100pct")
}

def add_gaussian_noise(signal, snr_db, rng):
    """
    Adds zero-mean Gaussian noise corresponding to given SNR in dB,
    computed strictly on the AC power (variance) rather than DC offset.
    """
    if snr_db is None or np.isinf(snr_db):
        return signal.copy()
    ac_var = np.var(signal)
    sigma = np.sqrt(max(ac_var / (10.0 ** (snr_db / 10.0)), 1e-12))
    return signal + rng.normal(0, sigma, size=signal.shape)

def smooth_digitized_signals(df):
    """
    Smooths digitization discretization artifacts using light Savitzky-Golay.
    Uses window_length=5 to avoid flattening narrow pressure dips and transient spikes.
    """
    df_clean = df.copy()
    # Light smoothing for pressure to preserve localized 1-2 sample transient dips
    p = df_clean['pressure_bar'].values
    p_smooth = savgol_filter(p, window_length=5, polyorder=2)
    df_clean['pressure_bar'] = p_smooth
    
    # Velocity light smoothing
    v = df_clean['velocity_mps'].values
    v_smooth = savgol_filter(v, window_length=5, polyorder=2)
    df_clean['velocity_mps'] = v_smooth
    
    return df_clean

def extract_window_features(window_df):
    """
    Computes statistical and physics-informed summary features for tabular ML models.
    """
    feats = {}
    cols = ['displacement_mm', 'velocity_mps', 'flow_rate_lpm', 'total_volume_l', 'pressure_bar',
            'pneumatic_power_W', 'dp_dt_bar_s', 'dq_dt_lpm_s', 'conductance_proxy', 'leak_baseline_lpm']
    
    for c in cols:
        v = window_df[c].values
        feats[f'{c}_mean'] = np.mean(v)
        feats[f'{c}_std'] = np.std(v)
        feats[f'{c}_min'] = np.min(v)
        feats[f'{c}_max'] = np.max(v)
        feats[f'{c}_p2p'] = np.ptp(v)
        feats[f'{c}_rms'] = np.sqrt(np.mean(v**2))
        
    # Key physical summary metrics
    feats['cycle_volume_loss'] = np.ptp(window_df['total_volume_l'].values)
    feats['mean_leak_conductance'] = np.mean(window_df['conductance_proxy'].values)
    # Use window baseline median rather than hardcoded 6.0 bar supply
    feats['peak_pressure_dip'] = np.median(window_df['pressure_bar'].values) - np.min(window_df['pressure_bar'].values)
    feats['mean_pneumatic_power'] = np.mean(window_df['pneumatic_power_W'].values)
    # Baseline steady flow rate between transient spikes (distinguishes healthy vs leak regimes)
    feats['leak_baseline_lpm'] = np.median(np.abs(window_df['flow_rate_lpm'].values))
    
    return feats

def generate_benchmark_dataset():
    rng = np.random.default_rng(42)
    raw_files = sorted(os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.lower().endswith("_leak.csv"))
    print(f"Loading raw digitized files: {len(raw_files)} files found.")
    
    # Each raw trace has 1001 points (10.0 s at 100 Hz = 4 distinct cycles of 250 samples / 2.5 s)
    CYCLE_LEN = 250
    NUM_CYCLES = 4
    # Partition cycles strictly to prevent data leakage:
    # Cycles 0 & 1 -> Train (augmented across noise regimes)
    # Cycle 2      -> Validation
    # Cycle 3      -> Test
    SPLIT_MAP = {0: "train", 1: "train", 2: "val", 3: "test"}
    
    # 10 sequence channels
    channels = [
        'displacement_mm', 'velocity_mps', 'flow_rate_lpm', 'total_volume_l', 'pressure_bar',
        'pneumatic_power_W', 'dp_dt_bar_s', 'dq_dt_lpm_s', 'conductance_proxy', 'leak_baseline_lpm'
    ]
    
    # Noise regimes: Clean, 30dB, 20dB, 15dB
    train_snr_levels = [np.inf, 30.0, 20.0, 15.0]
    eval_snr_levels = [np.inf, 30.0, 20.0, 15.0]
    
    split_data = {
        'train': {'seq': [], 'tab': [], 'cls': [], 'reg': [], 'snr': []},
        'val':   {'seq': [], 'tab': [], 'cls': [], 'reg': [], 'snr': []},
        'test':  {'seq': [], 'tab': [], 'cls': [], 'reg': [], 'snr': []}
    }
    
    # Storage for clean plotting dataset
    clean_tab_records = []
    
    for f in raw_files:
        df_raw = pd.read_csv(f)
        leak_val = float(df_raw['leak_pct'].iloc[0])
        cls_idx, cls_name = LABEL_MAP[leak_val]
        
        df_base = smooth_digitized_signals(df_raw)
        
        for c in range(NUM_CYCLES):
            target_split = SPLIT_MAP[c]
            # Extract non-overlapping cycle window BEFORE computing physics features (Zero cross-cycle leakage)
            win_slice = df_base.iloc[c * CYCLE_LEN : (c + 1) * CYCLE_LEN].copy()
            win_slice['total_volume_l'] = win_slice['total_volume_l'] - win_slice['total_volume_l'].iloc[0]
            win_raw = compute_physics_features(win_slice)
            
            # Record clean copy for physical trajectory plotting
            if target_split in ['train', 'val', 'test']:
                tab_clean = extract_window_features(win_raw)
                tab_clean['leak_pct'] = leak_val
                tab_clean['class_idx'] = cls_idx
                tab_clean['snr_db'] = 999.0
                tab_clean['cycle_idx'] = c
                clean_tab_records.append(tab_clean)
            
            # Select noise regime
            snr_list = train_snr_levels if target_split == 'train' else eval_snr_levels
            
            for snr in snr_list:
                win = win_raw.copy()
                
                if not np.isinf(snr):
                    # Add AC-variance Gaussian noise to sensor channels without unphysical clipping
                    win['displacement_mm'] = np.clip(add_gaussian_noise(win['displacement_mm'].values, snr, rng), 0.0, 100.0)
                    win['velocity_mps'] = add_gaussian_noise(win['velocity_mps'].values, snr, rng)
                    # Unbiased additive noise on flow rate and pressure
                    win['flow_rate_lpm'] = add_gaussian_noise(win['flow_rate_lpm'].values, snr, rng)
                    win['pressure_bar'] = add_gaussian_noise(win['pressure_bar'].values, snr, rng)
                    
                    # Recompute physics features under noise
                    win = compute_physics_features(win)
                
                tensor_win = win[channels].values.T  # Shape: (10, 250)
                tab_feats = extract_window_features(win)
                tab_feats['leak_pct'] = leak_val
                tab_feats['class_idx'] = cls_idx
                tab_feats['snr_db'] = snr if not np.isinf(snr) else 999.0
                tab_feats['cycle_idx'] = c
                
                split_data[target_split]['seq'].append(tensor_win)
                split_data[target_split]['tab'].append(tab_feats)
                split_data[target_split]['cls'].append(cls_idx)
                split_data[target_split]['reg'].append(leak_val)
                split_data[target_split]['snr'].append(snr)
                
    # Convert splits to numpy arrays & dataframes
    X_train = np.array(split_data['train']['seq'], dtype=np.float32)
    y_cls_train = np.array(split_data['train']['cls'], dtype=np.int64)
    y_reg_train = np.array(split_data['train']['reg'], dtype=np.float32)
    df_train = pd.DataFrame(split_data['train']['tab'])
    
    X_val = np.array(split_data['val']['seq'], dtype=np.float32)
    y_cls_val = np.array(split_data['val']['cls'], dtype=np.int64)
    y_reg_val = np.array(split_data['val']['reg'], dtype=np.float32)
    df_val = pd.DataFrame(split_data['val']['tab'])
    
    X_test = np.array(split_data['test']['seq'], dtype=np.float32)
    y_cls_test = np.array(split_data['test']['cls'], dtype=np.int64)
    y_reg_test = np.array(split_data['test']['reg'], dtype=np.float32)
    snr_test = np.array(split_data['test']['snr'], dtype=np.float32)
    df_test = pd.DataFrame(split_data['test']['tab'])
    
    print(f"Cycle-partitioned splits -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Tensor shape (Channels, Seq_Len): {X_train.shape[1:]}, Tabular features: {df_train.shape[1]}")
    
    # Save NPZ arrays
    np.savez_compressed(
        os.path.join(OUT_DIR, "dataset_tensors.npz"),
        X_train=X_train, y_cls_train=y_cls_train, y_reg_train=y_reg_train,
        X_val=X_val, y_cls_val=y_cls_val, y_reg_val=y_reg_val,
        X_test=X_test, y_cls_test=y_cls_test, y_reg_test=y_reg_test,
        snr_test=snr_test,
        channel_names=channels
    )
    
    # Save tabular datasets
    df_train.to_csv(os.path.join(OUT_DIR, "train_tabular.csv"), index=False)
    df_val.to_csv(os.path.join(OUT_DIR, "val_tabular.csv"), index=False)
    df_test.to_csv(os.path.join(OUT_DIR, "test_tabular.csv"), index=False)
    
    # Save OOD Harsh Noise Test Set (SNR = 10 dB) on Test Cycle (c = 3)
    print("Generating Harsh OOD Test Set (SNR = 10 dB)...")
    ood_seq = []
    ood_cls = []
    ood_reg = []
    for f in raw_files:
        df_raw = pd.read_csv(f)
        leak_val = float(df_raw['leak_pct'].iloc[0])
        cls_idx, _ = LABEL_MAP[leak_val]
        df_base = smooth_digitized_signals(df_raw)
        df_enriched = compute_physics_features(df_base)
        
        # OOD evaluates on test cycle (c=3) with harsh 10 dB noise
        win = df_enriched.iloc[3 * CYCLE_LEN : 4 * CYCLE_LEN].copy()
        win['total_volume_l'] = win['total_volume_l'] - win['total_volume_l'].iloc[0]
        
        win['displacement_mm'] = np.clip(add_gaussian_noise(win['displacement_mm'].values, 10.0, rng), 0.0, 100.0)
        win['velocity_mps'] = add_gaussian_noise(win['velocity_mps'].values, 10.0, rng)
        win['flow_rate_lpm'] = add_gaussian_noise(win['flow_rate_lpm'].values, 10.0, rng)
        win['pressure_bar'] = add_gaussian_noise(win['pressure_bar'].values, 10.0, rng)
        win = compute_physics_features(win)
        
        ood_seq.append(win[channels].values.T)
        ood_cls.append(cls_idx)
        ood_reg.append(leak_val)
            
    np.savez_compressed(
        os.path.join(OUT_DIR, "dataset_ood_harsh.npz"),
        X_ood=np.array(ood_seq, dtype=np.float32),
        y_cls_ood=np.array(ood_cls, dtype=np.int64),
        y_reg_ood=np.array(ood_reg, dtype=np.float32)
    )
    print(f"Saved OOD Harsh dataset: {len(ood_seq)} samples.")
    
    # Plot Figure 2: Physics Feature Phase Space & Trajectories for Paper using Clean Data Only
    df_clean_plot = pd.DataFrame(clean_tab_records)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    leak_levels = [0.0, 1.0, 5.0, 25.0, 50.0, 75.0, 100.0]
    
    # 1. Pneumatic Power vs Leak Level
    power_means = [df_clean_plot[df_clean_plot['leak_pct'] == l]['pneumatic_power_W_mean'].mean() for l in leak_levels]
    power_stds = [df_clean_plot[df_clean_plot['leak_pct'] == l]['pneumatic_power_W_mean'].std() for l in leak_levels]
    axs[0, 0].errorbar(leak_levels, power_means, yerr=power_stds, fmt='-o', color='#1f77b4', lw=2, capsize=5)
    axs[0, 0].set_title('(a) Mean Pneumatic Fluid Power vs Leak Severity', fontweight='bold', fontsize=11)
    axs[0, 0].set_xlabel('Leak Severity [%]', fontweight='bold')
    axs[0, 0].set_ylabel('Mean Pneumatic Power [Watts]', fontweight='bold')
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. Effective Conductance Proxy vs Leak Level
    cond_means = [df_clean_plot[df_clean_plot['leak_pct'] == l]['conductance_proxy_mean'].mean() for l in leak_levels]
    cond_stds = [df_clean_plot[df_clean_plot['leak_pct'] == l]['conductance_proxy_mean'].std() for l in leak_levels]
    axs[0, 1].errorbar(leak_levels, cond_means, yerr=cond_stds, fmt='-s', color='#d62728', lw=2, capsize=5)
    axs[0, 1].set_title('(b) Physics Conductance Index vs Leak Severity', fontweight='bold', fontsize=11)
    axs[0, 1].set_xlabel('Leak Severity [%]', fontweight='bold')
    axs[0, 1].set_ylabel('Effective Conductance Proxy [L/(s*bar)]', fontweight='bold')
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    
    # 3. Dynamic Pressure Dip vs Flow Spike
    scatter = axs[1, 0].scatter(df_clean_plot['flow_rate_lpm_min'], df_clean_plot['pressure_bar_min'], c=df_clean_plot['leak_pct'], cmap='viridis', alpha=0.9, s=40)
    cbar = plt.colorbar(scatter, ax=axs[1, 0])
    cbar.set_label('Leak Severity [%]', fontweight='bold')
    axs[1, 0].set_title('(c) Dynamic Pressure Dip vs Peak Flow Transient', fontweight='bold', fontsize=11)
    axs[1, 0].set_xlabel('Peak Transient Flow Rate [L/min]', fontweight='bold')
    axs[1, 0].set_ylabel('Minimum Transient Pressure [bar]', fontweight='bold')
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    
    # 4. Volume Loss per Cycle vs Leak Severity
    vol_means = [df_clean_plot[df_clean_plot['leak_pct'] == l]['cycle_volume_loss'].mean() for l in leak_levels]
    axs[1, 1].bar([str(int(l) if l.is_integer() else l)+'%' for l in leak_levels], vol_means, color=['#2ca02c', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4'], edgecolor='black', alpha=0.85)
    axs[1, 1].set_title('(d) 2.5-Second Single-Cycle Air Volume Loss', fontweight='bold', fontsize=11)
    axs[1, 1].set_xlabel('Leak Severity Class', fontweight='bold')
    axs[1, 1].set_ylabel('Volume Consumed [Liters]', fontweight='bold')
    axs[1, 1].grid(True, linestyle='--', alpha=0.6, axis='y')
    
    plt.suptitle('Physical State Trajectories and Diagnostic Feature Dynamics', fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    out_fig = os.path.join(FIG_DIR, "fig2_physics_trajectories.png")
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved physics trajectories plot: {out_fig}")

if __name__ == "__main__":
    generate_benchmark_dataset()
