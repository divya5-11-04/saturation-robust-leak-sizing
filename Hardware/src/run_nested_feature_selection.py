"""Nested feature-set selection (Hardware). Put in Hardware/src/.
Outer: 5 seeds x 5-fold StratifiedGroupKFold (grouped by run). Inner: 3-fold grouped CV on the outer-train
runs only, choosing among 5 candidate sets by MAE. ExtraTrees (inner and outer: 200 trees, sqrt features).
Usage: python run_nested_feature_selection.py [path/to/Hardware]"""
import sys, json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base / "src"))
from hardware_features import load_hardware_dataset

df, fc = load_hardware_dataset(base / "hardware_dataset")
d = df[df["segment_idx"] == 0].reset_index(drop=True)
assert len(d) == 105
y, g, cc = d["leak_pct"].values, d["group_id"].values, d["coarse_class"].values
p9 = ["p_mean", "p_std", "p_min", "p_max", "p_range", "p_q10", "p_q90", "p_slope", "p_energy"]
cfg = {"Flow-12": [c for c in fc if c.startswith("q_")], "Generic-18": fc[:18], "Pressure-9": p9,
       "Pressure+Timing-14": p9 + ["num_strokes", "stroke_freq", "stroke_period", "period_std", "period_cv"],
       "Combined-33": list(fc)}
assert [len(v) for v in cfg.values()] == [12, 18, 9, 14, 33]

def fit_pred(cols, tr, te, seed):
    X = d[cols].values
    sc = StandardScaler().fit(X[tr])
    m = ExtraTreesRegressor(n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=-1)
    m.fit(sc.transform(X[tr]), y[tr])
    return np.clip(m.predict(sc.transform(X[te])), 0, 100)

SEEDS = [42, 100, 2024, 7, 999]
maes, picks = [], []
for s in SEEDS:
    oof = np.zeros(len(d))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(d, cc, groups=g):
        inner = list(StratifiedGroupKFold(3, shuffle=True, random_state=s).split(d.iloc[tr], cc[tr], groups=g[tr]))
        score = {n: np.mean([mean_absolute_error(y[tr][iv], fit_pred(c, tr[it], tr[iv], s)) for it, iv in inner])
                 for n, c in cfg.items()}
        best = min(score, key=score.get); picks.append(best)
        oof[te] = fit_pred(cfg[best], tr, te, s)
    maes.append(mean_absolute_error(y, oof)); print(f"seed {s}: MAE {maes[-1]:.3f}", flush=True)
sd = np.std(maes, ddof=1); ci = stats.t.ppf(0.975, len(maes) - 1) * sd / np.sqrt(len(maes))
cnt = Counter(picks)
print(f"Nested MAE {np.mean(maes):.2f} +/- {ci:.2f} (95% CI over {len(maes)} seeds; SD {sd:.2f})")
print("Chosen set counts (of 25 outer folds):", dict(cnt))
out = base / "results" / "metrics" / "nested_selection.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"seeds": SEEDS, "mae_per_seed": maes, "mean": float(np.mean(maes)), "ci95": float(ci), "chosen": dict(cnt)}, indent=2))
