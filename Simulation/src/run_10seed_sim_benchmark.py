"""
Multi-Seed Benchmark and Physics Loss Ablation in Simulation (Festo FluidSIM)
Runs 10 random seeds across:
1. PI-DTNet (Full model with mass continuity and thermodynamic monotonicity loss)
2. Ablation 1 (PI-DTNet architecture without physics loss terms)
3. Ablation 2 (PI-DTNet without physics-engineered feature channels, raw 5 channels)
Computes paired t-test, Wilcoxon signed-rank test, and OOD test robustness.
Saves results to Simulation/results/metrics/sim_10seed_physics_ablation.json
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from scipy import stats

# Ensure imports from current dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_and_evaluate import load_data, train_pidtnet, evaluate_pytorch_model, DEVICE

SEEDS = [42, 10, 20, 30, 40, 50, 60, 70, 80, 90]
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "metrics")
os.makedirs(OUT_DIR, exist_ok=True)

def run_10seed_benchmark():
    print(f"Loading simulation dataset on {DEVICE}...")
    data = load_data()
    
    results = {
        'seeds': SEEDS,
        'pidtnet_phys': [],
        'pidtnet_nophys': [],
        'pidtnet_no_phys_feats': [],
        'ood_pidtnet_phys': [],
        'ood_pidtnet_nophys': [],
        'acc_phys': [],
        'acc_nophys': []
    }
    
    print("\n" + "="*70)
    print("STARTING 10-SEED SIMULATION BENCHMARK: PHYSICS LOSS ABLATION")
    print("="*70)
    
    for idx, s in enumerate(SEEDS, 1):
        print(f"\n--- SEED {s} ({idx}/{len(SEEDS)}) ---")
        
        # 1. PI-DTNet (Full with physics loss)
        torch.manual_seed(s)
        np.random.seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        m_phys, _ = train_pidtnet(data, use_physics_loss=True, in_channels=10, epochs=25, batch_size=32, lr=1e-3)
        ev_phys = evaluate_pytorch_model(m_phys, data['X_test'], data['y_cls_test'], data['y_reg_test'], in_channels=10)
        ood_phys = evaluate_pytorch_model(m_phys, data['X_ood'], data['y_cls_ood'], data['y_reg_ood'], in_channels=10)
        
        # 2. Ablation 1 (Without physics loss)
        torch.manual_seed(s)
        np.random.seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        m_nophys, _ = train_pidtnet(data, use_physics_loss=False, in_channels=10, epochs=25, batch_size=32, lr=1e-3)
        ev_nophys = evaluate_pytorch_model(m_nophys, data['X_test'], data['y_cls_test'], data['y_reg_test'], in_channels=10)
        ood_nophys = evaluate_pytorch_model(m_nophys, data['X_ood'], data['y_cls_ood'], data['y_reg_ood'], in_channels=10)
        
        # 3. Ablation 2 (Without physics features - 5 channels)
        torch.manual_seed(s)
        np.random.seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        m_nofeat, _ = train_pidtnet(data, use_physics_loss=False, in_channels=5, epochs=25, batch_size=32, lr=1e-3)
        ev_nofeat = evaluate_pytorch_model(m_nofeat, data['X_test'], data['y_cls_test'], data['y_reg_test'], in_channels=5)
        
        results['pidtnet_phys'].append(ev_phys['MAE_pct'])
        results['pidtnet_nophys'].append(ev_nophys['MAE_pct'])
        results['pidtnet_no_phys_feats'].append(ev_nofeat['MAE_pct'])
        results['ood_pidtnet_phys'].append(ood_phys['MAE_pct'])
        results['ood_pidtnet_nophys'].append(ood_nophys['MAE_pct'])
        results['acc_phys'].append(ev_phys['Accuracy'])
        results['acc_nophys'].append(ev_nophys['Accuracy'])
        
        print(f"Seed {s:2d} | Test MAE: Phys={ev_phys['MAE_pct']:.2f}%, NoPhys={ev_nophys['MAE_pct']:.2f}% (Diff={ev_nophys['MAE_pct'] - ev_phys['MAE_pct']:+.2f} pp) | OOD MAE: Phys={ood_phys['MAE_pct']:.2f}%, NoPhys={ood_nophys['MAE_pct']:.2f}%")

    phys_arr = np.array(results['pidtnet_phys'])
    nophys_arr = np.array(results['pidtnet_nophys'])
    nofeat_arr = np.array(results['pidtnet_no_phys_feats'])
    ood_phys_arr = np.array(results['ood_pidtnet_phys'])
    ood_nophys_arr = np.array(results['ood_pidtnet_nophys'])
    
    acc_phys_arr = np.array(results['acc_phys'])
    acc_nophys_arr = np.array(results['acc_nophys'])
    t_acc, p_acc = stats.ttest_rel(acc_phys_arr, acc_nophys_arr)
    t_nf, p_nf = stats.ttest_rel(nofeat_arr, nophys_arr)

    diff_test = nophys_arr - phys_arr
    t_stat_test, p_val_test = stats.ttest_rel(nophys_arr, phys_arr)
    w_stat_test, w_pval_test = stats.wilcoxon(diff_test)
    
    diff_ood = ood_nophys_arr - ood_phys_arr
    t_stat_ood, p_val_ood = stats.ttest_rel(ood_nophys_arr, ood_phys_arr)
    w_stat_ood, w_pval_ood = stats.wilcoxon(diff_ood)
    
    summary = {
        'seeds': SEEDS,
        'pidtnet_phys_mae_per_seed': [float(x) for x in phys_arr],
        'pidtnet_nophys_mae_per_seed': [float(x) for x in nophys_arr],
        'pidtnet_nofeat_mae_per_seed': [float(x) for x in nofeat_arr],
        'test_phys_mae_mean': float(phys_arr.mean()),
        'test_phys_mae_std': float(phys_arr.std(ddof=1)),
        'test_nophys_mae_mean': float(nophys_arr.mean()),
        'test_nophys_mae_std': float(nophys_arr.std(ddof=1)),
        'test_nofeat_mae_mean': float(nofeat_arr.mean()),
        'test_nofeat_mae_std': float(nofeat_arr.std(ddof=1)),
        'test_diff_mean_pp': float(diff_test.mean()),
        'acc_phys_per_seed': [float(x) for x in acc_phys_arr],
        'acc_nophys_per_seed': [float(x) for x in acc_nophys_arr],
        'acc_phys_mean': float(acc_phys_arr.mean()),
        'acc_phys_std': float(acc_phys_arr.std(ddof=1)),
        'acc_nophys_mean': float(acc_nophys_arr.mean()),
        'acc_nophys_std': float(acc_nophys_arr.std(ddof=1)),
        'acc_t_stat': float(t_acc),
        'acc_p_val_paired_t': float(p_acc),
        'nofeat_vs_nophys_t_stat': float(t_nf),
        'nofeat_vs_nophys_p_val_paired_t': float(p_nf),
        'test_t_stat': float(t_stat_test),
        'test_p_val_paired_t': float(p_val_test),
        'test_wilcoxon_stat': float(w_stat_test),
        'test_wilcoxon_p_val': float(w_pval_test),
        'ood_phys_mae_mean': float(ood_phys_arr.mean()),
        'ood_phys_mae_std': float(ood_phys_arr.std(ddof=1)),
        'ood_nophys_mae_mean': float(ood_nophys_arr.mean()),
        'ood_nophys_mae_std': float(ood_nophys_arr.std(ddof=1)),
        'ood_diff_mean_pp': float(diff_ood.mean()),
        'ood_t_stat': float(t_stat_ood),
        'ood_p_val_paired_t': float(p_val_ood),
        'ood_wilcoxon_stat': float(w_stat_ood),
        'ood_wilcoxon_p_val': float(w_pval_ood)
    }
    
    out_json = os.path.join(OUT_DIR, "sim_10seed_physics_ablation.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=4)
        
    out_csv = os.path.join(OUT_DIR, "sim_10seed_physics_ablation.csv")
    df_res = pd.DataFrame({
        'seed': SEEDS,
        'PI_DTNet_with_phys_loss': phys_arr,
        'Ablation1_no_phys_loss': nophys_arr,
        'Ablation2_no_phys_features': nofeat_arr,
        'OOD_with_phys_loss': ood_phys_arr,
        'OOD_no_phys_loss': ood_nophys_arr,
        'Acc_with_phys_loss': acc_phys_arr,
        'Acc_no_phys_loss': acc_nophys_arr
    })
    df_res.to_csv(out_csv, index=False)
    
    print("\n" + "="*70)
    print("=== FINAL 10-SEED SIMULATION PHYSICS ABLATION SUMMARY ===")
    print("="*70)
    print(f"In-Distribution Test MAE:")
    print(f"  - PI-DTNet (with physics loss)  : {phys_arr.mean():.2f} +/- {phys_arr.std(ddof=1):.2f}%")
    print(f"  - Ablation 1 (w/o physics loss) : {nophys_arr.mean():.2f} +/- {nophys_arr.std(ddof=1):.2f}%")
    print(f"  - Ablation 2 (w/o phys features): {nofeat_arr.mean():.2f} +/- {nofeat_arr.std(ddof=1):.2f}%")
    print(f"  - Physics Loss Benefit          : {diff_test.mean():.2f} pp reduction")
    print(f"  - Paired t-test: t = {t_stat_test:.3f}, p = {p_val_test:.4e}")
    print(f"  - Wilcoxon test: W = {w_stat_test}, p = {w_pval_test:.4e}")
    print(f"  - No physics features vs no physics loss: t = {t_nf:.3f}, p = {p_nf:.4f}")
    print(f"\nClassification accuracy (fraction):")
    print(f"  - with physics loss : {acc_phys_arr.mean():.3f} +/- {acc_phys_arr.std(ddof=1):.3f}")
    print(f"  - w/o physics loss  : {acc_nophys_arr.mean():.3f} +/- {acc_nophys_arr.std(ddof=1):.3f}")
    print(f"  - Paired t-test: t = {t_acc:.3f}, p = {p_acc:.4f}")
    print(f"\nOut-of-Distribution (OOD Harsh SNR=10dB) Test MAE:")
    print(f"  - PI-DTNet (with physics loss)  : {ood_phys_arr.mean():.2f} +/- {ood_phys_arr.std(ddof=1):.2f}%")
    print(f"  - Ablation 1 (w/o physics loss) : {ood_nophys_arr.mean():.2f} +/- {ood_nophys_arr.std(ddof=1):.2f}%")
    print(f"  - OOD Physics Loss Benefit      : {diff_ood.mean():.2f} pp reduction")
    print(f"  - OOD Paired t-test: t = {t_stat_ood:.3f}, p = {p_val_ood:.4e}")
    print(f"Saved results to:\n  {out_json}\n  {out_csv}")

if __name__ == "__main__":
    run_10seed_benchmark()