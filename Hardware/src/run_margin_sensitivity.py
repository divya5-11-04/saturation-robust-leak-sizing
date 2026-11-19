"""PINN-HW sensitivity to the deadband margins and loss weights (14-feature PINN, 10 seeds x 5-fold StratifiedGroupKFold).
One setting is varied at a time; the others stay at the paper defaults (delta_p=0.10 bar, delta_tau=0.10 s, delta_y=0.02,
gamma_1=1.5, gamma_2=0.8). Reports mean +/- SD of the cross-validated MAE over seeds, next to ExtraTrees-14 on the same splits.
This is a sensitivity analysis on the evaluation folds. It does not show how the defaults were originally chosen.
Usage: python run_margin_sensitivity.py [path/to/Hardware]   -> results/metrics/margin_sensitivity.csv"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
from hardware_features import load_hardware_dataset
from train_hardware_models import PINNHWNet

df, _ = load_hardware_dataset(BASE / "hardware_dataset")
d = df[df["segment_idx"] == 0].reset_index(drop=True)
p9 = ["p_mean", "p_std", "p_min", "p_max", "p_range", "p_q10", "p_q90", "p_slope", "p_energy"]
f14 = p9 + ["num_strokes", "stroke_freq", "stroke_period", "period_std", "period_cv"]
X = d[f14].values; y = d["leak_pct"].values; c = d["class_17"].values; g = d["group_id"].values; co = d["coarse_class"].values
pm = d["p_mean"].values; sp = d["stroke_period"].values
SEEDS = [42, 100, 2024, 7, 999, 123, 777, 314, 888, 555]
DEF = dict(dp=0.10, dt=0.10, dy=0.02, g1=1.5, g2=0.8)

def pinn_mae(seed, dp, dt, dy, g1, g2):
    oof = np.zeros(len(d))
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(d, co, groups=g):
        sc = StandardScaler(); xt = torch.tensor(sc.fit_transform(X[tr]), dtype=torch.float32)
        xv = torch.tensor(sc.transform(X[va]), dtype=torch.float32)
        yt = torch.tensor(y[tr] / 100.0, dtype=torch.float32); ct = torch.tensor(c[tr], dtype=torch.long)
        p = torch.tensor(pm[tr], dtype=torch.float32); t = torch.tensor(sp[tr], dtype=torch.float32)
        dtau = t.unsqueeze(1) - t.unsqueeze(0); dpr = p.unsqueeze(1) - p.unsqueeze(0)
        torch.manual_seed(seed); m = PINNHWNet(in_dim=14)
        opt = torch.optim.AdamW(m.parameters(), lr=0.01, weight_decay=1e-3); m.train()
        for _ in range(130):
            opt.zero_grad(); lg, rg = m(xt)
            dpred = rg.unsqueeze(1) - rg.unsqueeze(0)
            lk = torch.mean(F.relu(dtau - dt) * F.relu(-dpred - dy))
            lp = torch.mean(F.relu(-dpr - dp) * F.relu(-dpred - dy))
            loss = F.mse_loss(rg, yt) * 10.0 + F.cross_entropy(lg, ct) + g1 * lk + g2 * lp
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): oof[va] = m(xv)[1].numpy() * 100.0
    return mean_absolute_error(y, oof)

def et_mae(seed):
    oof = np.zeros(len(d))
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(d, co, groups=g):
        sc = StandardScaler().fit(X[tr])
        m = ExtraTreesRegressor(n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=-1).fit(sc.transform(X[tr]), y[tr])
        oof[va] = np.clip(m.predict(sc.transform(X[va])), 0, 100)
    return mean_absolute_error(y, oof)

grid = [("default", DEF)]
for k, vals in [("dp", [0.02, 0.05, 0.15, 0.20]), ("dt", [0.02, 0.05, 0.15, 0.20]), ("dy", [0.0, 0.01, 0.05]),
                ("g", [(0.5, 0.4), (1.0, 0.5), (2.0, 1.0), (2.5, 1.2)])]:
    for v in vals:
        cfg = dict(DEF)
        if k == "g": cfg["g1"], cfg["g2"] = v
        else: cfg[k] = v
        grid.append((f"{k}={v}", cfg))
rows = []
et = np.array([et_mae(s) for s in SEEDS]); print(f"ExtraTrees-14: {et.mean():.2f} +/- {et.std(ddof=1):.2f}", flush=True)
for name, cfg in grid:
    v = np.array([pinn_mae(s, **cfg) for s in SEEDS])
    rows.append({"setting": name, **cfg, "mae_mean": v.mean(), "mae_sd": v.std(ddof=1), "mae_min_seed": v.min(), "mae_max_seed": v.max()})
    print(f"{name:14s} MAE {v.mean():.2f} +/- {v.std(ddof=1):.2f}  (ET-14 {et.mean():.2f})", flush=True)
out = BASE / "results" / "metrics" / "margin_sensitivity.csv"; out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print("Saved", out)
