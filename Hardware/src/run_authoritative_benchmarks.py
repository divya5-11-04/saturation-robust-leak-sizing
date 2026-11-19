"""
Master Authoritative Hardware Benchmarking Suite (Multi-Core Accelerated)
Regenerates all hardware metrics, tables, statistical tests, and LOLO evaluations
under a unified, reproducible framework.

Guarantees:
1. Unified deadband ranking loss formulation for PINN-HW:
   L_kin   = (1/B^2) sum_{i,j} ReLU(tau_i - tau_j - delta_tau) * ReLU(y_hat_j - y_hat_i - delta_y)
   L_press = (1/B^2) sum_{i,j} ReLU(p_j - p_i - delta_p)       * ReLU(y_hat_j - y_hat_i - delta_y)
   with delta_tau = 0.1 s, delta_p = 0.1 bar, delta_y = 0.02 (2% severity), gamma_1 = 1.5, gamma_2 = 0.8.
2. Exact mathematical reconciliation for Table III: (30*Unsat + 75*Sat)/105 == Overall.
3. Exact paired significance tests across 10 seeds and 15 LOLO levels.
4. Complete physical monotonicity violation evaluation across all pairs.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier, RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.nn.functional as F

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir / "src"))

from hardware_features import load_hardware_dataset, LEAK_LEVELS, LEAK_TO_CLASS, CLASS_TO_LEAK

SEEDS = [42, 100, 2024, 7, 999, 123, 777, 314, 888, 555]

class PINNHWNet(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.cls = nn.Linear(16, 17)
        self.reg = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.cls(h), self.reg(h).squeeze(-1)

def reg_to_stage(pred_reg: np.ndarray) -> np.ndarray:
    stage = np.zeros(len(pred_reg), dtype=int)
    for i, p in enumerate(pred_reg):
        if p < 18.75: stage[i] = 0
        elif p < 43.75: stage[i] = 1
        elif p < 68.75: stage[i] = 2
        else: stage[i] = 3
    return stage

def ci95(arr: list[float] | np.ndarray) -> float:
    arr = np.asarray(arr)
    return float(2.262 * np.std(arr, ddof=1) / np.sqrt(len(arr)))

def compute_violation_rates(y_true, y_pred, p_vals, tau_vals, delta_p=0.01, delta_tau=0.01):
    n = len(y_true)
    press_viol, press_pairs = 0, 0
    kin_viol, kin_pairs = 0, 0
    
    for i in range(n):
        for j in range(n):
            if y_true[i] > y_true[j]:
                if (p_vals[j] - p_vals[i]) >= delta_p:
                    press_pairs += 1
                    if y_pred[i] < y_pred[j]: press_viol += 1
                if (tau_vals[i] - tau_vals[j]) >= delta_tau:
                    kin_pairs += 1
                    if y_pred[i] < y_pred[j]: kin_viol += 1
                    
    v_press = (press_viol / press_pairs * 100.0) if press_pairs > 0 else 0.0
    v_kin = (kin_viol / kin_pairs * 100.0) if kin_pairs > 0 else 0.0
    v_comp = ((press_viol + kin_viol) / (press_pairs + kin_pairs) * 100.0) if (press_pairs + kin_pairs) > 0 else 0.0
    return v_press, v_kin, v_comp

def main():
    t0_all = time.time()
    data_dir = base_dir / "hardware_dataset"
    out_dir = base_dir / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading hardware dataset...", flush=True)
    df_all, feat_cols = load_hardware_dataset(data_dir)
    df_runs = df_all[df_all["segment_idx"] == 0].reset_index(drop=True)
    assert len(df_runs) == 105, f"Expected 105 runs, got {len(df_runs)}"
    print(f"Loaded 105 runs with {len(feat_cols)} features.", flush=True)
    
    y_reg = df_runs["leak_pct"].values
    y_cls = df_runs["class_17"].values
    y_stage = reg_to_stage(y_reg)
    groups = df_runs["group_id"].values
    coarse = df_runs["coarse_class"].values
    p_mean = df_runs["p_mean"].values
    stroke_period = df_runs["stroke_period"].values
    
    # Feature subsets
    flow_12 = [c for c in feat_cols if c.startswith("q_")]
    gen_18 = feat_cols[:18]
    p_stats_9 = ["p_mean", "p_std", "p_min", "p_max", "p_range", "p_q10", "p_q90", "p_slope", "p_energy"]
    timing_5 = ["num_strokes", "stroke_freq", "stroke_period", "period_std", "period_cv"]
    p_timing_14 = p_stats_9 + timing_5
    comb_33 = feat_cols
    
    assert len(flow_12) == 12
    assert len(gen_18) == 18
    assert len(p_stats_9) == 9
    assert len(p_timing_14) == 14
    assert len(comb_33) == 33
    
    mask_sat = (y_reg >= 18.75)
    mask_unsat = (y_reg < 18.75)
    assert mask_sat.sum() == 75
    assert mask_unsat.sum() == 30
    
    # -------------------------------------------------------------
    # 1. TABLE II & TABLE IV: 10-Seed 5-Fold Stratified Group CV
    # -------------------------------------------------------------
    print("\n" + "="*80, flush=True)
    print("STEP 1: 10-Seed 5-Fold CV for Table II, Table III, Table IV", flush=True)
    print("="*80, flush=True)
    
    models_t2 = [
        "PINN-HW (Physics-Regularized)",
        "PINN-HW (Ablation w/o Phys)",
        "ExtraTrees (33 Feats)",
        "Random Forest (33 Feats)",
        "Gradient Boosting (33 Feats)",
        "MLP Baseline (33 Feats)",
        "ExtraTrees (14 Feats)"
    ]
    
    t2_records = {m: {"mae": [], "rmse": [], "r2": [], "res_625": [], "acc_17": [], "acc_stage": [], "d_mae": []} for m in models_t2}
    t4_records = {m: {"v_press": [], "v_kin": [], "v_comp": []} for m in models_t2}
    single_seed42 = {}
    
    models_t3 = [
        "Flow-Only Baseline (12 Feats)",
        "Generic Statistical Baseline (18 Feats)",
        "Pressure Statistics Alone (9 Feats)",
        "Pressure + Stroke Timing (14 Feats)",
        "Combined Multi-Modal Set (33 Feats)",
        "PINN-HW (14 Feats, Physics)",
        "PINN-HW (14 Feats, No Physics)",
        "PINN-HW (33 Feats, Physics)",
        "PINN-HW (33 Feats, No Physics)"
    ]
    t3_records = {m: {"unsat": [], "sat": [], "all": [], "r2_sat": []} for m in models_t3}
    
    for s_idx, seed in enumerate(SEEDS):
        t_seed_start = time.time()
        print(f"Running Seed {seed} ({s_idx+1}/{len(SEEDS)})...", end=" ", flush=True)
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(sgkf.split(df_runs, coarse, groups=groups))
        
        oof_preds = {m: np.zeros(105) for m in models_t2 + models_t3}
        oof_cls_preds = {m: np.zeros(105, dtype=int) for m in models_t2}
        
        for tr, val in splits:
            # Scalers for various feature sets
            sc33 = StandardScaler(); X33_tr = sc33.fit_transform(df_runs.loc[tr, comb_33].values); X33_va = sc33.transform(df_runs.loc[val, comb_33].values)
            sc14 = StandardScaler(); X14_tr = sc14.fit_transform(df_runs.loc[tr, p_timing_14].values); X14_va = sc14.transform(df_runs.loc[val, p_timing_14].values)
            sc9  = StandardScaler(); X9_tr  = sc9.fit_transform(df_runs.loc[tr, p_stats_9].values); X9_va  = sc9.transform(df_runs.loc[val, p_stats_9].values)
            sc18 = StandardScaler(); X18_tr = sc18.fit_transform(df_runs.loc[tr, gen_18].values); X18_va = sc18.transform(df_runs.loc[val, gen_18].values)
            sc12 = StandardScaler(); X12_tr = sc12.fit_transform(df_runs.loc[tr, flow_12].values); X12_va = sc12.transform(df_runs.loc[val, flow_12].values)
            
            y_tr, y_va = y_reg[tr], y_reg[val]
            c_tr, c_va = y_cls[tr], y_cls[val]
            p_tr, tau_tr = p_mean[tr], stroke_period[tr]
            
            # --- Tree Baselines (Accelerated with n_jobs=-1) ---
            et33_reg = ExtraTreesRegressor(n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=-1)
            et33_cls = ExtraTreesClassifier(n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=-1)
            et33_reg.fit(X33_tr, y_tr); et33_cls.fit(X33_tr, c_tr)
            oof_preds["ExtraTrees (33 Feats)"][val] = np.clip(et33_reg.predict(X33_va), 0, 100)
            oof_cls_preds["ExtraTrees (33 Feats)"][val] = et33_cls.predict(X33_va)
            
            et14_reg = ExtraTreesRegressor(n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=-1)
            et14_reg.fit(X14_tr, y_tr)
            oof_preds["ExtraTrees (14 Feats)"][val] = np.clip(et14_reg.predict(X14_va), 0, 100)
            
            rf33_reg = RandomForestRegressor(n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=-1)
            rf33_cls = RandomForestClassifier(n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=-1)
            rf33_reg.fit(X33_tr, y_tr); rf33_cls.fit(X33_tr, c_tr)
            oof_preds["Random Forest (33 Feats)"][val] = np.clip(rf33_reg.predict(X33_va), 0, 100)
            oof_cls_preds["Random Forest (33 Feats)"][val] = rf33_cls.predict(X33_va)
            
            gb33_reg = GradientBoostingRegressor(n_estimators=50, random_state=seed)
            gb33_reg.fit(X33_tr, y_tr)
            oof_preds["Gradient Boosting (33 Feats)"][val] = np.clip(gb33_reg.predict(X33_va), 0, 100)
            
            mlp_reg = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=150, random_state=seed, tol=1e-3)
            mlp_reg.fit(X33_tr, y_tr)
            oof_preds["MLP Baseline (33 Feats)"][val] = np.clip(mlp_reg.predict(X33_va), 0, 100)
            
            # --- Table III Tree Variants ---
            et12 = ExtraTreesRegressor(n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=-1); et12.fit(X12_tr, y_tr)
            oof_preds["Flow-Only Baseline (12 Feats)"][val] = np.clip(et12.predict(X12_va), 0, 100)
            
            et18 = ExtraTreesRegressor(n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=-1); et18.fit(X18_tr, y_tr)
            oof_preds["Generic Statistical Baseline (18 Feats)"][val] = np.clip(et18.predict(X18_va), 0, 100)
            
            et9 = ExtraTreesRegressor(n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=-1); et9.fit(X9_tr, y_tr)
            oof_preds["Pressure Statistics Alone (9 Feats)"][val] = np.clip(et9.predict(X9_va), 0, 100)
            
            oof_preds["Pressure + Stroke Timing (14 Feats)"][val] = oof_preds["ExtraTrees (14 Feats)"][val]
            oof_preds["Combined Multi-Modal Set (33 Feats)"][val] = oof_preds["ExtraTrees (33 Feats)"][val]
            
            # --- Neural PINN Models (33 Feats & 14 Feats) ---
            p_tr_t = torch.tensor(p_tr, dtype=torch.float32)
            tau_tr_t = torch.tensor(tau_tr, dtype=torch.float32)
            y_tr_t = torch.tensor(y_tr / 100.0, dtype=torch.float32)
            c_tr_t = torch.tensor(c_tr, dtype=torch.long)
            
            diff_tau = tau_tr_t.unsqueeze(1) - tau_tr_t.unsqueeze(0)
            diff_p = p_tr_t.unsqueeze(1) - p_tr_t.unsqueeze(0)
            
            # 1. PINN-HW 33 Feats (Physics)
            torch.manual_seed(seed)
            m_pinn33 = PINNHWNet(in_dim=33)
            opt_pinn33 = torch.optim.AdamW(m_pinn33.parameters(), lr=0.01, weight_decay=1e-3)
            x_tr_33 = torch.tensor(X33_tr, dtype=torch.float32)
            m_pinn33.train()
            for _ in range(130):
                opt_pinn33.zero_grad()
                lg, rg = m_pinn33(x_tr_33)
                diff_pred = rg.unsqueeze(1) - rg.unsqueeze(0)
                l_kin = torch.mean(F.relu(diff_tau - 0.1) * F.relu(-diff_pred - 0.02))
                l_press = torch.mean(F.relu(-diff_p - 0.1) * F.relu(-diff_pred - 0.02))
                loss = F.mse_loss(rg, y_tr_t) * 10.0 + F.cross_entropy(lg, c_tr_t) + 1.5 * l_kin + 0.8 * l_press
                loss.backward(); opt_pinn33.step()
            m_pinn33.eval()
            with torch.no_grad():
                lg_va, rg_va = m_pinn33(torch.tensor(X33_va, dtype=torch.float32))
                oof_preds["PINN-HW (Physics-Regularized)"][val] = rg_va.numpy() * 100.0
                oof_cls_preds["PINN-HW (Physics-Regularized)"][val] = torch.argmax(lg_va, dim=1).numpy()
                oof_preds["PINN-HW (33 Feats, Physics)"][val] = rg_va.numpy() * 100.0
                
            # 2. PINN-HW 33 Feats (No Physics Ablation)
            torch.manual_seed(seed)
            m_abl33 = PINNHWNet(in_dim=33)
            opt_abl33 = torch.optim.AdamW(m_abl33.parameters(), lr=0.01, weight_decay=1e-3)
            m_abl33.train()
            for _ in range(130):
                opt_abl33.zero_grad()
                lg, rg = m_abl33(x_tr_33)
                loss = F.mse_loss(rg, y_tr_t) * 10.0 + F.cross_entropy(lg, c_tr_t)
                loss.backward(); opt_abl33.step()
            m_abl33.eval()
            with torch.no_grad():
                lg_va, rg_va = m_abl33(torch.tensor(X33_va, dtype=torch.float32))
                oof_preds["PINN-HW (Ablation w/o Phys)"][val] = rg_va.numpy() * 100.0
                oof_cls_preds["PINN-HW (Ablation w/o Phys)"][val] = torch.argmax(lg_va, dim=1).numpy()
                oof_preds["PINN-HW (33 Feats, No Physics)"][val] = rg_va.numpy() * 100.0
                
            # 3. PINN-HW 14 Feats (Physics)
            torch.manual_seed(seed)
            m_pinn14 = PINNHWNet(in_dim=14)
            opt_pinn14 = torch.optim.AdamW(m_pinn14.parameters(), lr=0.01, weight_decay=1e-3)
            x_tr_14 = torch.tensor(X14_tr, dtype=torch.float32)
            m_pinn14.train()
            for _ in range(130):
                opt_pinn14.zero_grad()
                lg, rg = m_pinn14(x_tr_14)
                diff_pred = rg.unsqueeze(1) - rg.unsqueeze(0)
                l_kin = torch.mean(F.relu(diff_tau - 0.1) * F.relu(-diff_pred - 0.02))
                l_press = torch.mean(F.relu(-diff_p - 0.1) * F.relu(-diff_pred - 0.02))
                loss = F.mse_loss(rg, y_tr_t) * 10.0 + F.cross_entropy(lg, c_tr_t) + 1.5 * l_kin + 0.8 * l_press
                loss.backward(); opt_pinn14.step()
            m_pinn14.eval()
            with torch.no_grad():
                _, rg_va = m_pinn14(torch.tensor(X14_va, dtype=torch.float32))
                oof_preds["PINN-HW (14 Feats, Physics)"][val] = rg_va.numpy() * 100.0
                
            # 4. PINN-HW 14 Feats (No Physics)
            torch.manual_seed(seed)
            m_abl14 = PINNHWNet(in_dim=14)
            opt_abl14 = torch.optim.AdamW(m_abl14.parameters(), lr=0.01, weight_decay=1e-3)
            m_abl14.train()
            for _ in range(130):
                opt_abl14.zero_grad()
                lg, rg = m_abl14(x_tr_14)
                loss = F.mse_loss(rg, y_tr_t) * 10.0 + F.cross_entropy(lg, c_tr_t)
                loss.backward(); opt_abl14.step()
            m_abl14.eval()
            with torch.no_grad():
                _, rg_va = m_abl14(torch.tensor(X14_va, dtype=torch.float32))
                oof_preds["PINN-HW (14 Feats, No Physics)"][val] = rg_va.numpy() * 100.0
                
        # End of 5 folds for this seed.
        for m in models_t2:
            pred = oof_preds[m]
            mae = mean_absolute_error(y_reg, pred)
            rmse = np.sqrt(mean_squared_error(y_reg, pred))
            r2 = r2_score(y_reg, pred)
            res625 = np.mean(np.abs(y_reg - pred) <= 6.25) * 100.0
            
            d_true = 4.0 * np.sqrt(np.clip(y_reg, 0, 100) / 100.0)
            d_pred = 4.0 * np.sqrt(np.clip(pred, 0, 100) / 100.0)
            d_mae = mean_absolute_error(d_true, d_pred)
            
            if m in oof_cls_preds and oof_cls_preds[m].sum() > 0:
                acc_17 = accuracy_score(y_cls, oof_cls_preds[m]) * 100.0
                acc_st = accuracy_score(y_stage, reg_to_stage(pred)) * 100.0
            else:
                acc_17 = 0.0
                acc_st = accuracy_score(y_stage, reg_to_stage(pred)) * 100.0
                
            t2_records[m]["mae"].append(mae)
            t2_records[m]["rmse"].append(rmse)
            t2_records[m]["r2"].append(r2)
            t2_records[m]["res_625"].append(res625)
            t2_records[m]["acc_17"].append(acc_17)
            t2_records[m]["acc_stage"].append(acc_st)
            t2_records[m]["d_mae"].append(d_mae)
            
            vp, vk, vc = compute_violation_rates(y_reg, pred, p_mean, stroke_period)
            t4_records[m]["v_press"].append(vp)
            t4_records[m]["v_kin"].append(vk)
            t4_records[m]["v_comp"].append(vc)
            
            if seed == 42:
                single_seed42[m] = {
                    "mae": mae, "rmse": rmse, "r2": r2, "res_625": res625, "acc_17": acc_17, "acc_stage": acc_st, "d_mae": d_mae
                }

        for m in models_t3:
            pred = oof_preds[m]
            m_unsat = mean_absolute_error(y_reg[mask_unsat], pred[mask_unsat])
            m_sat = mean_absolute_error(y_reg[mask_sat], pred[mask_sat])
            m_all = mean_absolute_error(y_reg, pred)
            r2_s = r2_score(y_reg[mask_sat], pred[mask_sat])
            
            weighted_all = (30.0 * m_unsat + 75.0 * m_sat) / 105.0
            assert abs(weighted_all - m_all) < 1e-12
            
            t3_records[m]["unsat"].append(m_unsat)
            t3_records[m]["sat"].append(m_sat)
            t3_records[m]["all"].append(m_all)
            t3_records[m]["r2_sat"].append(r2_s)
            
        print(f"done ({time.time() - t_seed_start:.1f}s)", flush=True)

    # -------------------------------------------------------------
    # 2. COMPILE TABLE II SUMMARY & SIGNIFICANCE TESTS
    # -------------------------------------------------------------
    print("\n" + "="*80, flush=True)
    print("TABLE II: 10-SEED HARDWARE BENCHMARKS (Mean ± 95% CI)", flush=True)
    print("="*80, flush=True)
    
    t2_rows = []
    for m in models_t2:
        m_mae = np.mean(t2_records[m]["mae"])
        ci_mae = ci95(t2_records[m]["mae"])
        s_mae = single_seed42[m]["mae"]
        m_rmse = np.mean(t2_records[m]["rmse"])
        m_r2 = np.mean(t2_records[m]["r2"])
        m_res = np.mean(t2_records[m]["res_625"])
        m_stg = np.mean(t2_records[m]["acc_stage"])
        m_dmae = np.mean(t2_records[m]["d_mae"])
        
        t2_rows.append({
            "Model": m,
            "10-Seed MAE (%)": f"{m_mae:.2f} ± {ci_mae:.2f}",
            "Single Seed-42 MAE (%)": f"{s_mae:.2f}",
            "RMSE (%)": f"{m_rmse:.2f}",
            "R2": f"{m_r2:.4f}",
            "1-Turn Res (%)": f"{m_res:.1f}",
            "4-Stage Acc (%)": f"{m_stg:.1f}",
            "Orifice Error (mm)": f"{m_dmae:.3f}",
            "raw_mean_mae": m_mae,
            "raw_ci_mae": ci_mae
        })
        print(f"{m:<32} | {m_mae:5.2f} ± {ci_mae:4.2f} | Seed-42: {s_mae:5.2f} | RMSE: {m_rmse:5.2f} | R2: {m_r2:.4f} | Res: {m_res:4.1f}% | Stg: {m_stg:4.1f}% | d_err: {m_dmae:.3f} mm", flush=True)

    df_t2 = pd.DataFrame(t2_rows)
    df_t2.to_csv(out_dir / "table_II_authoritative.csv", index=False)

    pinn_mae = np.array(t2_records["PINN-HW (Physics-Regularized)"]["mae"])
    abl_mae  = np.array(t2_records["PINN-HW (Ablation w/o Phys)"]["mae"])
    et33_mae = np.array(t2_records["ExtraTrees (33 Feats)"]["mae"])
    et14_mae = np.array(t2_records["ExtraTrees (14 Feats)"]["mae"])
    
    t_pinn_abl, p_pinn_abl = stats.ttest_rel(pinn_mae, abl_mae)
    w_pinn_abl = stats.wilcoxon(pinn_mae, abl_mae)
    
    t_pinn_et33, p_pinn_et33 = stats.ttest_rel(pinn_mae, et33_mae)
    t_pinn_et14, p_pinn_et14 = stats.ttest_rel(pinn_mae, et14_mae)
    t_et14_et33, p_et14_et33 = stats.ttest_rel(et14_mae, et33_mae)
    w_et14_et33 = stats.wilcoxon(et14_mae, et33_mae)
    
    indist_tests = {
        "pinn_vs_abl": {"diff": float(np.mean(abl_mae - pinn_mae)), "t": float(t_pinn_abl), "p": float(p_pinn_abl), "w_p": float(w_pinn_abl.pvalue)},
        "pinn_vs_et33": {"diff": float(np.mean(et33_mae - pinn_mae)), "t": float(t_pinn_et33), "p": float(p_pinn_et33)},
        "pinn_vs_et14": {"diff": float(np.mean(et14_mae - pinn_mae)), "t": float(t_pinn_et14), "p": float(p_pinn_et14)},
        "et14_vs_et33": {"diff": float(np.mean(et33_mae - et14_mae)), "t": float(t_et14_et33), "p": float(p_et14_et33), "w_p": float(w_et14_et33.pvalue)}
    }
    with open(out_dir / "in_distribution_tests_authoritative.json", "w") as f:
        json.dump(indist_tests, f, indent=2)

    print("\nIn-Distribution Paired Tests (10 seeds):", flush=True)
    print(f"PINN-HW vs Ablation:  diff = {indist_tests['pinn_vs_abl']['diff']:+.3f} pp, t = {t_pinn_abl:.3f}, p = {p_pinn_abl:.4f}", flush=True)
    print(f"PINN-HW vs ET-33:     diff = {indist_tests['pinn_vs_et33']['diff']:+.3f} pp, t = {t_pinn_et33:.3f}, p = {p_pinn_et33:.4f}", flush=True)
    print(f"PINN-HW vs ET-14:     diff = {indist_tests['pinn_vs_et14']['diff']:+.3f} pp, t = {t_pinn_et14:.3f}, p = {p_pinn_et14:.4f}", flush=True)
    print(f"ET-14 vs ET-33:       diff = {indist_tests['et14_vs_et33']['diff']:+.3f} pp, t = {t_et14_et33:.3f}, p = {p_et14_et33:.4f}", flush=True)

    # -------------------------------------------------------------
    # 3. COMPILE TABLE III SUMMARY & RECONCILIATION
    # -------------------------------------------------------------
    print("\n" + "="*80, flush=True)
    print("TABLE III: SENSOR SATURATION FEATURE ABLATION (10-Seed Reconciliation)", flush=True)
    print("="*80, flush=True)
    
    t3_rows = []
    for m in models_t3:
        u_mean = np.mean(t3_records[m]["unsat"])
        u_ci   = ci95(t3_records[m]["unsat"])
        s_mean = np.mean(t3_records[m]["sat"])
        s_ci   = ci95(t3_records[m]["sat"])
        a_mean = np.mean(t3_records[m]["all"])
        a_ci   = ci95(t3_records[m]["all"])
        r2_s   = np.mean(t3_records[m]["r2_sat"])
        
        weighted = (30.0 * u_mean + 75.0 * s_mean) / 105.0
        assert abs(weighted - a_mean) < 1e-4
        
        t3_rows.append({
            "Configuration": m,
            "Unsaturated MAE (<18.75%)": f"{u_mean:.2f} ± {u_ci:.2f}",
            "Saturated MAE (>=18.75%)": f"{s_mean:.2f} ± {s_ci:.2f}",
            "Overall MAE (105 Runs)": f"{a_mean:.2f} ± {a_ci:.2f}",
            "Saturated R2": f"{r2_s:.4f}",
            "raw_u": round(u_mean, 3),
            "raw_s": round(s_mean, 3),
            "raw_a": round(a_mean, 3),
            "weighted_recon": round(weighted, 3)
        })
        print(f"{m:<40} | Unsat: {u_mean:5.2f} ± {u_ci:4.2f} | Sat: {s_mean:5.2f} ± {s_ci:4.2f} | Overall: {a_mean:5.2f} ± {a_ci:4.2f} | Recon: {weighted:5.2f} | R2_sat: {r2_s:.3f}", flush=True)

    df_t3 = pd.DataFrame(t3_rows)
    df_t3.to_csv(out_dir / "table_III_authoritative.csv", index=False)

    # -------------------------------------------------------------
    # 4. COMPILE TABLE IV: PHYSICAL MONOTONICITY VIOLATIONS
    # -------------------------------------------------------------
    print("\n" + "="*80, flush=True)
    print("TABLE IV: PHYSICAL MONOTONICITY VIOLATIONS (10 Seeds, Mean ± Sample SD & 95% CI)", flush=True)
    print("="*80, flush=True)
    
    t4_rows = []
    for m in models_t2:
        vp_m, vp_sd, vp_ci = np.mean(t4_records[m]["v_press"]), np.std(t4_records[m]["v_press"], ddof=1), ci95(t4_records[m]["v_press"])
        vk_m, vk_sd, vk_ci = np.mean(t4_records[m]["v_kin"]),   np.std(t4_records[m]["v_kin"], ddof=1),   ci95(t4_records[m]["v_kin"])
        vc_m, vc_sd, vc_ci = np.mean(t4_records[m]["v_comp"]),  np.std(t4_records[m]["v_comp"], ddof=1),  ci95(t4_records[m]["v_comp"])
        
        t4_rows.append({
            "Model": m,
            "Pressure Violation (%)": f"{vp_m:.2f} ± {vp_sd:.2f} (CI ±{vp_ci:.2f})",
            "Kinematic Violation (%)": f"{vk_m:.2f} ± {vk_sd:.2f} (CI ±{vk_ci:.2f})",
            "Composite Violation (%)": f"{vc_m:.2f} ± {vc_sd:.2f} (CI ±{vc_ci:.2f})",
            "raw_comp_mean": vc_m,
            "raw_comp_sd": vc_sd
        })
        print(f"{m:<32} | Press: {vp_m:5.2f} ± {vp_sd:4.2f} | Kin: {vk_m:5.2f} ± {vk_sd:4.2f} | Comp: {vc_m:5.2f} ± {vc_sd:4.2f} (CI ±{vc_ci:.2f})", flush=True)

    df_t4 = pd.DataFrame(t4_rows)
    df_t4.to_csv(out_dir / "table_IV_authoritative.csv", index=False)

    vc_pinn = np.array(t4_records["PINN-HW (Physics-Regularized)"]["v_comp"])
    vc_abl  = np.array(t4_records["PINN-HW (Ablation w/o Phys)"]["v_comp"])
    t_viol, p_viol = stats.ttest_rel(vc_pinn, vc_abl)
    w_viol = stats.wilcoxon(vc_pinn, vc_abl)
    print(f"\nPINN-HW Phys vs NoPhys Composite Violations:", flush=True)
    print(f"  Nominal Drop: {np.mean(vc_abl) - np.mean(vc_pinn):.2f} pp | Paired t-test t = {t_viol:.3f}, p = {p_viol:.4f} | Wilcoxon p = {w_viol.pvalue:.4f}", flush=True)
    pd.DataFrame({"seed": SEEDS, "v_comp_pinn": vc_pinn, "v_comp_abl": vc_abl}).to_csv(out_dir / "violations_per_seed.csv", index=False)
    d_v = vc_pinn - vc_abl
    t_nb_v = d_v.mean() / (d_v.std(ddof=1) * np.sqrt(1/10 + 1/4))
    p_nb_v = 2 * stats.t.sf(abs(t_nb_v), df=len(d_v) - 1)
    print(f"  Nadeau-Bengio: t = {t_nb_v:.3f}, p = {p_nb_v:.4f}", flush=True)

    # -------------------------------------------------------------
    # 5. STEP 2: LEAVE-ONE-LEVEL-OUT (LOLO) BENCHMARK ACROSS 15 LEVELS
    # -------------------------------------------------------------
    print("\n" + "="*80, flush=True)
    print("STEP 2: Leave-One-Level-Out (LOLO) Cross-Validation Across 15 Levels (10 Seeds)", flush=True)
    print("="*80, flush=True)
    
    intermediate_levels = [l for l in LEAK_LEVELS if 0.0 < l < 100.0]
    assert len(intermediate_levels) == 15
    
    lolo_models = ["PINN-HW", "PINN-HW-NoPhys", "ExtraTrees-33", "ExtraTrees-14", "RandomForest-33", "Generic-ExtraTrees-18"]
    lolo_level_means = {m: [] for m in lolo_models}
    lolo_level_dmae = {m: [] for m in lolo_models}
    
    for lvl_idx, held_out in enumerate(intermediate_levels):
        t_lvl_start = time.time()
        test_mask = (y_reg == held_out)
        train_mask = ~test_mask
        n_test = test_mask.sum()
        d_true = 4.0 * np.sqrt(held_out / 100.0)
        
        print(f"Held-out {held_out:5.2f}% (Level {lvl_idx+1}/15, n={n_test})...", end=" ", flush=True)
        
        y_train, y_test = y_reg[train_mask], y_reg[test_mask]
        c_train, c_test = y_cls[train_mask], y_cls[test_mask]
        p_train, tau_train = p_mean[train_mask], stroke_period[train_mask]
        
        seed_maes = {m: [] for m in lolo_models}
        seed_dmaes = {m: [] for m in lolo_models}
        
        for seed in SEEDS:
            sc33 = StandardScaler(); X33_tr = sc33.fit_transform(df_runs.loc[train_mask, comb_33].values); X33_te = sc33.transform(df_runs.loc[test_mask, comb_33].values)
            sc14 = StandardScaler(); X14_tr = sc14.fit_transform(df_runs.loc[train_mask, p_timing_14].values); X14_te = sc14.transform(df_runs.loc[test_mask, p_timing_14].values)
            sc18 = StandardScaler(); X18_tr = sc18.fit_transform(df_runs.loc[train_mask, gen_18].values); X18_te = sc18.transform(df_runs.loc[test_mask, gen_18].values)
            
            # ET-33
            et33 = ExtraTreesRegressor(n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=-1); et33.fit(X33_tr, y_train)
            p_et33 = np.clip(et33.predict(X33_te), 0, 100)
            seed_maes["ExtraTrees-33"].append(mean_absolute_error(y_test, p_et33))
            seed_dmaes["ExtraTrees-33"].append(mean_absolute_error(np.full_like(p_et33, d_true), 4.0 * np.sqrt(p_et33 / 100.0)))
            
            # ET-14
            et14 = ExtraTreesRegressor(n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=-1); et14.fit(X14_tr, y_train)
            p_et14 = np.clip(et14.predict(X14_te), 0, 100)
            seed_maes["ExtraTrees-14"].append(mean_absolute_error(y_test, p_et14))
            seed_dmaes["ExtraTrees-14"].append(mean_absolute_error(np.full_like(p_et14, d_true), 4.0 * np.sqrt(p_et14 / 100.0)))
            
            # RF-33
            rf33 = RandomForestRegressor(n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=-1); rf33.fit(X33_tr, y_train)
            p_rf33 = np.clip(rf33.predict(X33_te), 0, 100)
            seed_maes["RandomForest-33"].append(mean_absolute_error(y_test, p_rf33))
            seed_dmaes["RandomForest-33"].append(mean_absolute_error(np.full_like(p_rf33, d_true), 4.0 * np.sqrt(p_rf33 / 100.0)))
            
            # Generic ET-18
            et18 = ExtraTreesRegressor(n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=-1); et18.fit(X18_tr, y_train)
            p_et18 = np.clip(et18.predict(X18_te), 0, 100)
            seed_maes["Generic-ExtraTrees-18"].append(mean_absolute_error(y_test, p_et18))
            seed_dmaes["Generic-ExtraTrees-18"].append(mean_absolute_error(np.full_like(p_et18, d_true), 4.0 * np.sqrt(p_et18 / 100.0)))
            
            # PINN-HW (Physics)
            x_tr_t = torch.tensor(X33_tr, dtype=torch.float32)
            y_tr_t = torch.tensor(y_train / 100.0, dtype=torch.float32)
            c_tr_t = torch.tensor(c_train, dtype=torch.long)
            p_tr_t = torch.tensor(p_train, dtype=torch.float32)
            tau_tr_t = torch.tensor(tau_train, dtype=torch.float32)
            
            diff_tau = tau_tr_t.unsqueeze(1) - tau_tr_t.unsqueeze(0)
            diff_p = p_tr_t.unsqueeze(1) - p_tr_t.unsqueeze(0)
            
            torch.manual_seed(seed)
            m_pinn = PINNHWNet(in_dim=33)
            opt_pinn = torch.optim.AdamW(m_pinn.parameters(), lr=0.01, weight_decay=1e-3)
            m_pinn.train()
            for _ in range(130):
                opt_pinn.zero_grad()
                lg, rg = m_pinn(x_tr_t)
                diff_pred = rg.unsqueeze(1) - rg.unsqueeze(0)
                l_kin = torch.mean(F.relu(diff_tau - 0.1) * F.relu(-diff_pred - 0.02))
                l_press = torch.mean(F.relu(-diff_p - 0.1) * F.relu(-diff_pred - 0.02))
                loss = F.mse_loss(rg, y_tr_t) * 10.0 + F.cross_entropy(lg, c_tr_t) + 1.5 * l_kin + 0.8 * l_press
                loss.backward(); opt_pinn.step()
            m_pinn.eval()
            with torch.no_grad():
                _, rg_te = m_pinn(torch.tensor(X33_te, dtype=torch.float32))
                p_pinn = rg_te.numpy() * 100.0
            seed_maes["PINN-HW"].append(mean_absolute_error(y_test, p_pinn))
            seed_dmaes["PINN-HW"].append(mean_absolute_error(np.full_like(p_pinn, d_true), 4.0 * np.sqrt(np.clip(p_pinn, 0, 100) / 100.0)))
            
            # PINN-HW No Physics Ablation
            torch.manual_seed(seed)
            m_abl = PINNHWNet(in_dim=33)
            opt_abl = torch.optim.AdamW(m_abl.parameters(), lr=0.01, weight_decay=1e-3)
            m_abl.train()
            for _ in range(130):
                opt_abl.zero_grad()
                lg, rg = m_abl(x_tr_t)
                loss = F.mse_loss(rg, y_tr_t) * 10.0 + F.cross_entropy(lg, c_tr_t)
                loss.backward(); opt_abl.step()
            m_abl.eval()
            with torch.no_grad():
                _, rg_te = m_abl(torch.tensor(X33_te, dtype=torch.float32))
                p_abl = rg_te.numpy() * 100.0
            seed_maes["PINN-HW-NoPhys"].append(mean_absolute_error(y_test, p_abl))
            seed_dmaes["PINN-HW-NoPhys"].append(mean_absolute_error(np.full_like(p_abl, d_true), 4.0 * np.sqrt(np.clip(p_abl, 0, 100) / 100.0)))
            
        for m in lolo_models:
            lolo_level_means[m].append(np.mean(seed_maes[m]))
            lolo_level_dmae[m].append(np.mean(seed_dmaes[m]))
            
        print(f"done ({time.time() - t_lvl_start:.1f}s) -> PINN: {lolo_level_means['PINN-HW'][-1]:.2f}%, ET-14: {lolo_level_means['ExtraTrees-14'][-1]:.2f}%, ET-33: {lolo_level_means['ExtraTrees-33'][-1]:.2f}%", flush=True)

    print("\n" + "="*80, flush=True)
    print("LOLO OVERALL SUMMARY ACROSS 15 UNSEEN APERTURES (Mean over 15 Level Means)", flush=True)
    print("="*80, flush=True)
    
    lolo_summary_rows = []
    for m in lolo_models:
        mean_lolo = float(np.mean(lolo_level_means[m]))
        mean_d = float(np.mean(lolo_level_dmae[m]))
        worst_lolo = float(np.max(lolo_level_means[m]))
        worst_lvl = intermediate_levels[int(np.argmax(lolo_level_means[m]))]
        lolo_summary_rows.append({
            "Model": m,
            "Mean LOLO MAE (%)": round(mean_lolo, 2),
            "Mean Turn Error (rot)": round(mean_lolo / 6.25, 2),
            "Mean Orifice d_err (mm)": round(mean_d, 3),
            "Worst-Level MAE (%)": round(worst_lolo, 2),
            "Worst Level (%)": worst_lvl
        })
        print(f"{m:<24} | Mean LOLO MAE: {mean_lolo:5.2f}% | Orifice Error: {mean_d:.3f} mm | Worst: {worst_lolo:5.2f}% at {worst_lvl}%", flush=True)

    df_lolo_sum = pd.DataFrame(lolo_summary_rows)
    df_lolo_sum.to_csv(out_dir / "table_lolo_summary_authoritative.csv", index=False)

    p_pinn = np.array(lolo_level_means["PINN-HW"])
    p_abl  = np.array(lolo_level_means["PINN-HW-NoPhys"])
    p_et33 = np.array(lolo_level_means["ExtraTrees-33"])
    p_et14 = np.array(lolo_level_means["ExtraTrees-14"])
    
    t_lolo_abl,  p_lolo_abl  = stats.ttest_rel(p_pinn, p_abl)
    t_lolo_et33, p_lolo_et33 = stats.ttest_rel(p_pinn, p_et33)
    t_lolo_et14, p_lolo_et14 = stats.ttest_rel(p_pinn, p_et14)
    t_lolo_et_pair, p_lolo_et_pair = stats.ttest_rel(p_et14, p_et33)
    
    lolo_tests = {
        "pinn_vs_abl":  {"diff": float(np.mean(p_abl - p_pinn)),  "t": float(t_lolo_abl),  "p": float(p_lolo_abl)},
        "pinn_vs_et33": {"diff": float(np.mean(p_et33 - p_pinn)), "t": float(t_lolo_et33), "p": float(p_lolo_et33)},
        "pinn_vs_et14": {"diff": float(np.mean(p_et14 - p_pinn)), "t": float(t_lolo_et14), "p": float(p_lolo_et14)},
        "et14_vs_et33": {"diff": float(np.mean(p_et33 - p_et14)), "t": float(t_lolo_et_pair), "p": float(p_lolo_et_pair)}
    }
    with open(out_dir / "lolo_paired_tests_authoritative.json", "w") as f:
        json.dump(lolo_tests, f, indent=2)

    print("\nLOLO Paired Tests (n = 15 levels, df = 14):", flush=True)
    print(f"PINN-HW vs Ablation: diff = {lolo_tests['pinn_vs_abl']['diff']:+.3f} pp, t = {t_lolo_abl:.3f}, p = {p_lolo_abl:.4f}", flush=True)
    print(f"PINN-HW vs ET-33:    diff = {lolo_tests['pinn_vs_et33']['diff']:+.3f} pp, t = {t_lolo_et33:.3f}, p = {p_lolo_et33:.4f}", flush=True)
    print(f"PINN-HW vs ET-14:    diff = {lolo_tests['pinn_vs_et14']['diff']:+.3f} pp, t = {t_lolo_et14:.3f}, p = {p_lolo_et14:.4f}", flush=True)
    print(f"ET-14 vs ET-33:      diff = {lolo_tests['et14_vs_et33']['diff']:+.3f} pp, t = {t_lolo_et_pair:.3f}, p = {p_lolo_et_pair:.4f}", flush=True)
    
    print(f"\nAll Master Authoritative Hardware Benchmarks Complete in {time.time() - t0_all:.1f}s!", flush=True)

if __name__ == "__main__":
    main()
