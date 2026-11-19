"""
Sensor Saturation Ablation Benchmarks across 10 Seeds:
1. Flow-Only (12 feats)
2. Generic Statistical Baseline (18 feats)
3. Pressure Stats Alone (9 feats)
4. Pressure-Only with Stroke Timing (14 feats)
5. Combined Multi-Modal (33 feats)
6. PINN-HW on 14 Pressure Feats (with physics)
7. PINN-HW on 14 Pressure Feats (without physics ablation)
"""

import sys
import json
from pathlib import Path

# Absolute paths
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir / "src"))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn.functional as F

from hardware_features import load_hardware_dataset
from train_hardware_models import PINNHWNet

SEEDS = [42, 100, 2024, 7, 999, 123, 777, 314, 888, 555]

data_dir = base_dir / "hardware_dataset"
out_dir = base_dir / "results" / "metrics"
out_dir.mkdir(parents=True, exist_ok=True)

df_all, feat_cols = load_hardware_dataset(data_dir)
df_runs = df_all[df_all["segment_idx"] == 0].reset_index(drop=True)
assert len(df_runs) == 105
y_all = df_runs["leak_pct"].values
groups = df_runs["group_id"].values
c_all = df_runs["class_17"].values

mask_unsat = y_all < 18.75
mask_sat = y_all >= 18.75

# Feature definitions
flow_12 = [c for c in feat_cols if c.startswith("q_")]
gen_18 = feat_cols[:18]
p_stats_9 = ["p_mean", "p_std", "p_min", "p_max", "p_range", "p_q10", "p_q90", "p_slope", "p_energy"]
assert all(c in feat_cols for c in p_stats_9), "Missing pressure features!"
p_timing_14 = p_stats_9 + ["num_strokes", "stroke_freq", "stroke_period", "period_std", "period_cv"]
assert all(c in feat_cols for c in p_timing_14), "Missing timing features!"
combined_33 = feat_cols

assert len(flow_12) == 12
assert len(gen_18) == 18
assert len(p_stats_9) == 9
assert len(p_timing_14) == 14
assert len(combined_33) == 33

feature_configs = {
    "Flow-Only": flow_12,
    "Generic-18": gen_18,
    "Pressure-9": p_stats_9,
    "Pressure-14": p_timing_14,
    "Combined-33": combined_33
}

et_records = {k: {"sat": [], "unsat": [], "all": []} for k in feature_configs}

for seed in SEEDS:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_splits = list(sgkf.split(df_runs, df_runs["coarse_class"], groups=groups))
    
    for name, cols in feature_configs.items():
        X_sub = df_runs[cols].values
        oof = np.zeros(len(df_runs))
        for tr, val in cv_splits:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_sub[tr])
            X_va = sc.transform(X_sub[val])
            m = ExtraTreesRegressor(n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=1)
            m.fit(X_tr, y_all[tr])
            oof[val] = np.clip(m.predict(X_va), 0.0, 100.0)
            
        et_records[name]["sat"].append(mean_absolute_error(y_all[mask_sat], oof[mask_sat]))
        et_records[name]["unsat"].append(mean_absolute_error(y_all[mask_unsat], oof[mask_unsat]))
        et_records[name]["all"].append(mean_absolute_error(y_all, oof))

# PINN on 14 features (with physics AND without physics ablation)
pinn_14_phys = {"sat": [], "all": []}
pinn_14_nophys = {"sat": [], "all": []}

for seed in SEEDS:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_splits = list(sgkf.split(df_runs, df_runs["coarse_class"], groups=groups))
    
    for use_p, rec in [(True, pinn_14_phys), (False, pinn_14_nophys)]:
        oof_p = np.zeros(len(df_runs))
        for tr, val in cv_splits:
            sc = StandardScaler()
            X_tr = sc.fit_transform(df_runs.loc[tr, p_timing_14].values)
            X_va = sc.transform(df_runs.loc[val, p_timing_14].values)
            
            y_tr, c_tr = y_all[tr], c_all[tr]
            p_tr = df_runs.loc[tr, "p_mean"].values
            tau_tr = df_runs.loc[tr, "stroke_period"].values
            
            torch.manual_seed(seed)
            m = PINNHWNet(in_dim=14)
            opt = torch.optim.AdamW(m.parameters(), lr=0.01, weight_decay=1e-3)
            x_t = torch.tensor(X_tr, dtype=torch.float32)
            y_t = torch.tensor(y_tr / 100.0, dtype=torch.float32)
            c_t = torch.tensor(c_tr, dtype=torch.long)
            p_t = torch.tensor(p_tr, dtype=torch.float32)
            tau_t = torch.tensor(tau_tr, dtype=torch.float32)
            
            m.train()
            for epoch in range(130):
                opt.zero_grad()
                logits, pred = m(x_t)
                if use_p:
                    diff_tau = tau_t.unsqueeze(1) - tau_t.unsqueeze(0)
                    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
                    loss_kin = torch.mean(F.relu(diff_tau - 0.1) * F.relu(-diff_pred - 0.02))
                    diff_p = p_t.unsqueeze(1) - p_t.unsqueeze(0)
                    loss_press = torch.mean(F.relu(-diff_p - 0.1) * F.relu(-diff_pred - 0.02))
                    loss = F.mse_loss(pred, y_t) * 10.0 + F.cross_entropy(logits, c_t) + 1.5 * loss_kin + 0.8 * loss_press
                else:
                    loss = F.mse_loss(pred, y_t) * 10.0 + F.cross_entropy(logits, c_t)
                loss.backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                _, p_v = m(torch.tensor(X_va, dtype=torch.float32))
                oof_p[val] = p_v.numpy() * 100.0
        rec["sat"].append(mean_absolute_error(y_all[mask_sat], oof_p[mask_sat]))
        rec["all"].append(mean_absolute_error(y_all, oof_p))

# Statistical tests
t_gain, p_gain = stats.ttest_rel(et_records["Pressure-9"]["sat"], et_records["Pressure-14"]["sat"])
w_gain = stats.wilcoxon(et_records["Pressure-9"]["sat"], et_records["Pressure-14"]["sat"])

# Nadeau-Bengio corrected resampled t-test
diffs = np.array(et_records["Pressure-9"]["sat"]) - np.array(et_records["Pressure-14"]["sat"])
mean_diff = float(np.mean(diffs))
var_diff = float(np.var(diffs, ddof=1))
n_seeds = 10
n_test_ratio = 1.0 / 4.0 # 5-fold CV: test fraction = 0.2, train fraction = 0.8 -> 0.2/0.8 = 0.25
se_corrected = np.sqrt(var_diff * (1.0 / n_seeds + n_test_ratio))
t_nadeau_bengio = float(mean_diff / se_corrected)
p_nadeau_bengio = float(2.0 * (1.0 - stats.t.cdf(abs(t_nadeau_bengio), df=n_seeds - 1)))

# Test 14-feat PINN with vs without physics
t_pinn14_loss, p_pinn14_loss = stats.ttest_rel(pinn_14_phys["sat"], pinn_14_nophys["sat"])

def paired_overall(a, b):
    """Paired tests on per-seed overall MAE (a - b): uncorrected t, Nadeau-Bengio t, Wilcoxon."""
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    t, p = stats.ttest_rel(a, b)
    t_nb = float(d.mean() / (d.std(ddof=1) * np.sqrt(1.0 / len(d) + 0.25)))
    p_nb = float(2.0 * stats.t.sf(abs(t_nb), df=len(d) - 1))
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": float(d.mean()),
            "a_lower_in_seeds": int((d < 0).sum()), "t": float(t), "p": float(p),
            "t_nadeau_bengio": t_nb, "p_nadeau_bengio": p_nb, "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue),
            "a_per_seed": [round(float(x), 4) for x in a], "b_per_seed": [round(float(x), 4) for x in b]}

overall_tests = {
    "et14_vs_et33_overall": paired_overall(et_records["Pressure-14"]["all"], et_records["Combined-33"]["all"]),
    "et14_vs_pinn14_overall": paired_overall(et_records["Pressure-14"]["all"], pinn_14_phys["all"]),
    "pinn14_phys_vs_nophys_sat": paired_overall(pinn_14_phys["sat"], pinn_14_nophys["sat"]),
    "pinn14_phys_vs_nophys_overall": paired_overall(pinn_14_phys["all"], pinn_14_nophys["all"]),
}

summary = {
    "flow_12_sat": f"{np.mean(et_records['Flow-Only']['sat']):.2f} ± {2.262*np.std(et_records['Flow-Only']['sat'], ddof=1)/np.sqrt(10):.2f}",
    "gen_18_sat": f"{np.mean(et_records['Generic-18']['sat']):.2f} ± {2.262*np.std(et_records['Generic-18']['sat'], ddof=1)/np.sqrt(10):.2f}",
    "p_9_sat": f"{np.mean(et_records['Pressure-9']['sat']):.2f} ± {2.262*np.std(et_records['Pressure-9']['sat'], ddof=1)/np.sqrt(10):.2f}",
    "p_14_sat": f"{np.mean(et_records['Pressure-14']['sat']):.2f} ± {2.262*np.std(et_records['Pressure-14']['sat'], ddof=1)/np.sqrt(10):.2f}",
    "combined_33_sat": f"{np.mean(et_records['Combined-33']['sat']):.2f} ± {2.262*np.std(et_records['Combined-33']['sat'], ddof=1)/np.sqrt(10):.2f}",
    "pinn_14_phys_sat": f"{np.mean(pinn_14_phys['sat']):.2f} ± {2.262*np.std(pinn_14_phys['sat'], ddof=1)/np.sqrt(10):.2f}",
    "pinn_14_phys_all": f"{np.mean(pinn_14_phys['all']):.2f} ± {2.262*np.std(pinn_14_phys['all'], ddof=1)/np.sqrt(10):.2f}",
    "pinn_14_nophys_sat": f"{np.mean(pinn_14_nophys['sat']):.2f} ± {2.262*np.std(pinn_14_nophys['sat'], ddof=1)/np.sqrt(10):.2f}",
    "pinn_14_nophys_all": f"{np.mean(pinn_14_nophys['all']):.2f} ± {2.262*np.std(pinn_14_nophys['all'], ddof=1)/np.sqrt(10):.2f}",
    "p_9_vs_p_14_diff": mean_diff,
    "p_9_sat_per_seed": [round(float(x), 4) for x in et_records["Pressure-9"]["sat"]],
    "p_14_sat_per_seed": [round(float(x), 4) for x in et_records["Pressure-14"]["sat"]],
    "p_diff_per_seed": [round(float(d), 4) for d in diffs],
    "all_10_seeds_win": bool(all(d > 0 for d in diffs)),
    "t_gain": float(t_gain),
    "p_gain": float(p_gain),
    "w_gain_stat": float(w_gain.statistic),
    "w_gain_p": float(w_gain.pvalue),
    "t_nadeau_bengio": t_nadeau_bengio,
    "p_nadeau_bengio": p_nadeau_bengio,
    "pinn_14_phys_vs_nophys_sat_diff": float(np.mean(pinn_14_phys["sat"]) - np.mean(pinn_14_nophys["sat"])),
    "pinn_14_phys_vs_nophys_p": float(p_pinn14_loss),
    "overall_tests": overall_tests
}

out_file = out_dir / "saturation_ablation_summary.json"
with open(out_file, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nWritten summary to {out_file}:")
print(json.dumps(summary, indent=2))