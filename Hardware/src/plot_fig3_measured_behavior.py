"""Fig. 3: measured behavior versus leak setting (pressure sag, stroke period, flow saturation).
Save as Hardware/src/plot_fig3_measured_behavior.py. Reads Hardware/hardware_dataset/data_*.csv and writes
Hardware/results/figures/fig3_measured_behavior.png. Run:  python plot_fig3_measured_behavior.py
Optional: python plot_fig3_measured_behavior.py path/to/Hardware"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "figure.dpi": 300, "savefig.dpi": 300, "lines.linewidth": 1.6,
    "grid.alpha": 0.35, "grid.linestyle": "-",
})

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "hardware_dataset"
OUT_FIG = BASE / "results" / "figures" / "fig3_measured_behavior.png"

LEAK_LEVELS = [0.0, 6.25, 12.5, 18.75, 25.0, 31.25, 37.5, 43.75, 50.0, 56.25, 62.5, 68.75, 75.0, 81.25, 87.5, 93.75, 100.0]

def parse_leak_pct(stem: str) -> float:
    s = stem.replace("data_", "")
    if s == "12_50" or s.startswith("12_50_"):   # 12.5% files are named 12_50
        return 12.5
    for l in sorted(LEAK_LEVELS, reverse=True):
        r = str(l).replace(".", "_")
        if r.endswith("_0"):
            r = r[:-2]
        if s == r or s.startswith(r + "_"):
            return l
    raise ValueError(f"Could not parse leak from: {stem}")

records = []
files = sorted(DATA_DIR.glob("data_*.csv"))
for f in files:
    d = pd.read_csv(f)
    t, p, q = d["Elapsed_Time_sec"].values, d["Pressure_bar"].values, d["Airflow_LPM"].values
    p_std = max(float(np.std(p)), 0.005)
    peaks, _ = find_peaks(-p, distance=4, prominence=0.25 * p_std)   # pressure dips = strokes
    dur = float(t[-1] - t[0])
    freq = len(peaks) / dur
    tau = 1.0 / freq if freq > 0.1 else dur                           # no strokes -> full run duration
    records.append({"leak_pct": parse_leak_pct(f.stem), "p_mean": float(np.mean(p)),
                    "tau": tau, "sat": float(np.mean(q >= 15.8) * 100.0)})
runs = pd.DataFrame(records)
assert len(runs) == 105, f"expected 105 runs, found {len(runs)}"

g = runs.groupby("leak_pct").agg(n=("p_mean", "count"), p=("p_mean", "mean"), p_sd=("p_mean", "std"),
                                 tau=("tau", "mean"), tau_sd=("tau", "std"), sat=("sat", "mean")).reset_index()
g[["p_sd", "tau_sd"]] = g[["p_sd", "tau_sd"]].fillna(0.0)
x = g["leak_pct"].values
few = (g["n"] <= 3).values
many = ~few

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

def band(ax, col, sd, color, marker, ms_many, ms_few, label, ylabel, title):
    ax.fill_between(x, g[col] - g[sd], g[col] + g[sd], color=color, alpha=0.18)
    ax.errorbar(x, g[col], yerr=g[sd], fmt="-", color=color, ecolor=color, capsize=3.5, label=label)
    ax.scatter(x[many], g.loc[many, col], marker=marker, color=color, s=ms_many, zorder=5)
    ax.scatter(x[few], g.loc[few, col], marker=marker, facecolors="white", edgecolors=color, s=ms_few,
               linewidth=1.8, zorder=6, label="levels with 3 or fewer runs")
    ax.set_title(title); ax.set_xlabel("Leak setting (% of regulator opening)"); ax.set_ylabel(ylabel)
    ax.grid(True)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="upper left", framealpha=0.9)

band(axes[0], "p", "p_sd", "#1f77b4", "o", 40, 65, "Mean line pressure", "Line pressure (bar gauge)", "(a) Supply pressure sag")
band(axes[1], "tau", "tau_sd", "#d62728", "s", 35, 55, "Mean stroke period", r"Stroke period $\tau_{stroke}$ (s)", "(b) Stroke period dilation")

ax = axes[2]
ax.plot(x, g["sat"], "-^", color="#2ca02c", linewidth=1.8, markersize=6, label="Samples at 15.8 LPM or above (%)")
ax.axvline(18.75, color="black", linestyle="--", alpha=0.75, label="18.75% leak")
ax.set_title("(c) Flow sensor saturation"); ax.set_xlabel("Leak setting (% of regulator opening)")
ax.set_ylabel("Saturated samples (%)"); ax.set_ylim(-5, 105); ax.grid(True)
ax.legend(loc="lower right", framealpha=0.9)

plt.tight_layout()
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FIG, bbox_inches="tight")
print(g.round(3).to_string(index=False))
print(f"Saved figure to: {OUT_FIG}")