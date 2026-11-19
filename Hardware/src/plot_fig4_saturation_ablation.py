"""Fig. 4: unsaturated and saturated MAE by feature configuration (Table II in the paper).
Save as Hardware/src/plot_fig4_saturation_ablation.py. Reads Hardware/results/metrics/table_III_authoritative.csv
(written by run_authoritative_benchmarks.py) and writes Hardware/results/figures/fig4_saturation_ablation.png.
Run: python plot_fig4_saturation_ablation.py   (optional arg: path to the Hardware folder)"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
table = pd.read_csv(BASE / "results" / "metrics" / "table_III_authoritative.csv").set_index("Configuration")

# bottom-to-top order in the figure: csv row name -> label
rows = [
    ("PINN-HW (33 Feats, Physics)", "PINN-HW (33 features)"),
    ("PINN-HW (14 Feats, Physics)", "PINN-HW (14 features)"),
    ("Combined Multi-Modal Set (33 Feats)", "Combined (33)"),
    ("Pressure + Stroke Timing (14 Feats)", "Pressure and timing (14)"),
    ("Pressure Statistics Alone (9 Feats)", "Pressure statistics (9)"),
    ("Generic Statistical Baseline (18 Feats)", "Generic statistics (18)"),
    ("Flow-Only Baseline (12 Feats)", "Flow only (12)"),
]

def split(cell):
    mean, ci = str(cell).split("±")
    return float(mean), float(ci)

models = [label for _, label in rows]
unsat_mae, unsat_ci = map(np.array, zip(*[split(table.loc[r, "Unsaturated MAE (<18.75%)"]) for r, _ in rows]))
sat_mae, sat_ci = map(np.array, zip(*[split(table.loc[r, "Saturated MAE (>=18.75%)"]) for r, _ in rows]))

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "figure.dpi": 300, "savefig.dpi": 300})
y_pos = np.arange(len(models)); height = 0.35
fig, ax = plt.subplots(figsize=(5.2, 4.0))
kw = {"elinewidth": 1.0, "ecolor": "black", "capthick": 1.0}
ax.barh(y_pos + height / 2, unsat_mae, height, xerr=unsat_ci, capsize=2, color="#3c6e9f", label="Unsaturated (n = 30)", error_kw=kw)
ax.barh(y_pos - height / 2, sat_mae, height, xerr=sat_ci, capsize=2, color="#f58518", label="Saturated (n = 75)", error_kw=kw)
for i in range(len(models)):
    ax.text(unsat_mae[i] + unsat_ci[i] + 0.25, y_pos[i] + height / 2, f"{unsat_mae[i]:.2f}", va="center", ha="left", fontsize=7.5)
    ax.text(sat_mae[i] + sat_ci[i] + 0.25, y_pos[i] - height / 2, f"{sat_mae[i]:.2f}", va="center", ha="left", fontsize=7.5)
ax.set_yticks(y_pos); ax.set_yticklabels(models)
ax.set_xlabel("Mean absolute error (% of full leak range)"); ax.set_xlim(0, 24)
ax.xaxis.grid(True, linestyle="-", alpha=0.3, color="lightgray"); ax.set_axisbelow(True)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.2); ax.spines["bottom"].set_linewidth(1.2)
ax.legend(loc="lower right", frameon=False)
plt.tight_layout()
out = BASE / "results" / "figures" / "fig4_saturation_ablation.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=300)
print(pd.DataFrame({"model": models, "unsat": unsat_mae, "unsat_ci": unsat_ci, "sat": sat_mae, "sat_ci": sat_ci}).to_string(index=False))
print("Saved", out)