"""
Physical Law Consistency & Monotonicity Violation Analysis on Hardware (Festo MPS.pa)
Evaluates whether machine learning models adhere to foundational thermodynamic and kinematic laws:
1. Pressure Sag Monotonicity: Larger leak orifice -> increased diverted flow -> reduced line pressure (p_i <= p_j)
2. Kinematic Stroke Delay Monotonicity: Larger leak orifice -> reduced chamber charging rate -> increased stroke period (tau_i >= tau_j)

A physical violation occurs whenever a model's prediction inverts the physical law:
- Pressure Violation: When p_i < p_j and actual leak y_i > y_j, but model predicts y_hat_i < y_hat_j
- Kinematic Violation: When tau_i > tau_j and actual leak y_i > y_j, but model predicts y_hat_i < y_hat_j

Evaluated across:
- 10-Seed 5-Fold Cross-Validation (In-Distribution)
- 15 Leave-One-Level-Out (LOLO) Folds (Unseen Apertures)
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir / "src"))

from hardware_features import load_hardware_dataset, LEAK_LEVELS
from train_hardware_models import PINNHWNet, reg_to_stage

SEEDS = [42, 100, 2024, 7, 999, 123, 777, 314, 888, 555]
DEVICE = torch.device("cpu")

def compute_violation_rates(y_true, y_pred, p_vals, tau_vals, delta_p=0.01, delta_tau=0.01):
    """
    Computes pairwise physical violation rates for pairs (i, j) where y_true[i] > y_true[j].
    """
    n = len(y_true)
    press_violations = 0
    press_pairs = 0
    
    kin_violations = 0
    kin_pairs = 0
    
    for i in range(n):
        for j in range(n):
            if y_true[i] > y_true[j]:
                # Pressure test: if p_i < p_j (by at least delta_p bar)
                if (p_vals[j] - p_vals[i]) >= delta_p:
                    press_pairs += 1
                    if y_pred[i] < y_pred[j]:
                        press_violations += 1
                        
                # Kinematic test: if tau_i > tau_j (by at least delta_tau seconds)
                if (tau_vals[i] - tau_vals[j]) >= delta_tau:
                    kin_pairs += 1
                    if y_pred[i] < y_pred[j]:
                        kin_violations += 1
                        
    v_press = (press_violations / press_pairs * 100.0) if press_pairs > 0 else 0.0
    v_kin = (kin_violations / kin_pairs * 100.0) if kin_pairs > 0 else 0.0
    v_composite = ((press_violations + kin_violations) / (press_pairs + kin_pairs) * 100.0) if (press_pairs + kin_pairs) > 0 else 0.0
    
    return {
        'press_violation_pct': float(v_press),
        'kin_violation_pct': float(v_kin),
        'composite_violation_pct': float(v_composite),
        'press_pairs': int(press_pairs),
        'kin_pairs': int(kin_pairs)
    }

def run_evaluation():
    data_dir = base_dir / "hardware_dataset"
    out_dir = base_dir / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_all, feat_cols = load_hardware_dataset(data_dir)
    df_runs = df_all[df_all["segment_idx"] == 0].reset_index(drop=True)
    y_all = df_runs["leak_pct"].values
    groups = df_runs["group_id"].values
    
    p_mean = df_runs["p_mean"].values
    stroke_period = df_runs["stroke_period"].values
    
    print(f"Loaded {len(df_runs)} physical hardware runs across {len(np.unique(y_all))} leak levels.")
    
    # 1. 10-Seed CV Monotonicity Violation Benchmark
    print("\n" + "="*70)
    print("1. EVALUATING PHYSICAL MONOTONICITY VIOLATIONS ACROSS 10 CV SEEDS")
    print("="*70)
    
    models = ["ExtraTrees_33", "RandomForest_33", "MLP_33", "PINN_HW_Phys", "PINN_HW_NoPhys", "ExtraTrees_14"]
    cv_records = {m: {'v_press': [], 'v_kin': [], 'v_comp': []} for m in models}
    
    p_timing_14 = [c for c in feat_cols[:9]] + ["num_strokes", "stroke_freq", "stroke_period", "period_std", "period_cv"]
    
    for seed in SEEDS:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        cv_splits = list(sgkf.split(df_runs, df_runs["coarse_class"], groups=groups))
        
        oof_preds = {m: np.zeros(len(df_runs)) for m in models}
        
        for tr, val in cv_splits:
            sc33 = StandardScaler()
            X33_tr = sc33.fit_transform(df_runs[feat_cols].values[tr])
            X33_va = sc33.transform(df_runs[feat_cols].values[val])
            
            sc14 = StandardScaler()
            X14_tr = sc14.fit_transform(df_runs[p_timing_14].values[tr])
            X14_va = sc14.transform(df_runs[p_timing_14].values[val])
            
            # ExtraTrees 33
            et33 = ExtraTreesRegressor(n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=1)
            et33.fit(X33_tr, y_all[tr])
            oof_preds["ExtraTrees_33"][val] = np.clip(et33.predict(X33_va), 0.0, 100.0)
            
            # Random Forest 33
            rf33 = RandomForestRegressor(n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=1)
            rf33.fit(X33_tr, y_all[tr])
            oof_preds["RandomForest_33"][val] = np.clip(rf33.predict(X33_va), 0.0, 100.0)
            
            # MLP 33
            mlp33 = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=150, random_state=seed, tol=1e-3)
            mlp33.fit(X33_tr, y_all[tr])
            oof_preds["MLP_33"][val] = np.clip(mlp33.predict(X33_va), 0.0, 100.0)
            
            # ExtraTrees 14
            et14 = ExtraTreesRegressor(n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=1)
            et14.fit(X14_tr, y_all[tr])
            oof_preds["ExtraTrees_14"][val] = np.clip(et14.predict(X14_va), 0.0, 100.0)
            
            # PINN-HW Models (with and without physics)
            c_tr = df_runs["class_17"].values[tr]
            X_tr_t = torch.tensor(X33_tr, dtype=torch.float32)
            c_tr_t = torch.tensor(c_tr, dtype=torch.long)
            y_tr_t = torch.tensor(y_all[tr] / 100.0, dtype=torch.float32)  # scaled in float64 first, as in run_authoritative_benchmarks.py
            
            p_tr_t = torch.tensor(p_mean[tr], dtype=torch.float32)
            tau_tr_t = torch.tensor(stroke_period[tr], dtype=torch.float32)
            
            X_va_t = torch.tensor(X33_va, dtype=torch.float32)
            
            for use_p, key in [(True, "PINN_HW_Phys"), (False, "PINN_HW_NoPhys")]:
                torch.manual_seed(seed)
                net = PINNHWNet(in_dim=33)
                opt = torch.optim.AdamW(net.parameters(), lr=0.01, weight_decay=1e-3)
                
                net.train()
                for _ in range(130):
                    opt.zero_grad()
                    lg, rg = net(X_tr_t)
                    if use_p:
                        # Pairwise kinematic monotonicity
                        diff_tau = tau_tr_t.unsqueeze(1) - tau_tr_t.unsqueeze(0)
                        diff_pred = rg.unsqueeze(1) - rg.unsqueeze(0)
                        loss_kin = torch.mean(F.relu(diff_tau - 0.1) * F.relu(-diff_pred - 0.02))
                        
                        # Pairwise supply pressure sag consistency
                        diff_p = p_tr_t.unsqueeze(1) - p_tr_t.unsqueeze(0)
                        loss_press = torch.mean(F.relu(-diff_p - 0.1) * F.relu(-diff_pred - 0.02))
                        
                        # same operation order as run_authoritative_benchmarks.py (float-level reproducibility)
                        loss = F.mse_loss(rg, y_tr_t) * 10.0 + F.cross_entropy(lg, c_tr_t) + 1.5 * loss_kin + 0.8 * loss_press
                    else:
                        loss = F.mse_loss(rg, y_tr_t) * 10.0 + F.cross_entropy(lg, c_tr_t)
                        
                    loss.backward()
                    opt.step()
                    
                net.eval()
                with torch.no_grad():
                    _, pred_v = net(X_va_t)
                    oof_preds[key][val] = np.clip(pred_v.squeeze().numpy() * 100.0, 0.0, 100.0)
                    
        # Compute violations for this seed
        for m in models:
            met = compute_violation_rates(y_all, oof_preds[m], p_mean, stroke_period)
            cv_records[m]['v_press'].append(met['press_violation_pct'])
            cv_records[m]['v_kin'].append(met['kin_violation_pct'])
            cv_records[m]['v_comp'].append(met['composite_violation_pct'])
            
    summary = {}
    print(f"\n{'Model':<18} | {'Press Viol %':<16} | {'Kin Viol %':<16} | {'Composite Viol %':<16}")
    print("-" * 72)
    for m in models:
        vp_m, vp_s = np.mean(cv_records[m]['v_press']), np.std(cv_records[m]['v_press'], ddof=1)
        vk_m, vk_s = np.mean(cv_records[m]['v_kin']), np.std(cv_records[m]['v_kin'], ddof=1)
        vc_m, vc_s = np.mean(cv_records[m]['v_comp']), np.std(cv_records[m]['v_comp'], ddof=1)
        summary[m] = {
            'press_viol_mean': float(vp_m), 'press_viol_std': float(vp_s),
            'kin_viol_mean': float(vk_m), 'kin_viol_std': float(vk_s),
            'comp_viol_mean': float(vc_m), 'comp_viol_std': float(vc_s),
            'comp_per_seed': [float(x) for x in cv_records[m]['v_comp']]
        }
        print(f"{m:<18} | {vp_m:5.2f} +/- {vp_s:4.2f}% | {vk_m:5.2f} +/- {vk_s:4.2f}% | {vc_m:5.2f} +/- {vc_s:4.2f}%")
        
    t_viol, p_viol = stats.ttest_rel(cv_records['PINN_HW_Phys']['v_comp'], cv_records['PINN_HW_NoPhys']['v_comp'])
    w_viol = stats.wilcoxon(cv_records['PINN_HW_Phys']['v_comp'], cv_records['PINN_HW_NoPhys']['v_comp'])
    summary['paired_test_phys_vs_nophys'] = {
        'mean_diff': float(np.mean(cv_records['PINN_HW_Phys']['v_comp']) - np.mean(cv_records['PINN_HW_NoPhys']['v_comp'])),
        't_stat': float(t_viol),
        'p_val': float(p_viol),
        'w_stat': float(w_viol.statistic),
        'w_pval': float(w_viol.pvalue)
    }
    print(f"\nPaired t-test PINN_HW_Phys vs PINN_HW_NoPhys: t = {t_viol:.4f}, p = {p_viol:.4f}, Wilcoxon p = {w_viol.pvalue:.4f}")
    
    out_file = out_dir / "physical_monotonicity_violations.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved violation benchmark to {out_file}")

if __name__ == "__main__":
    run_evaluation()