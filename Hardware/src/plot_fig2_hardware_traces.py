"""Fig. 2: hardware traces for five leak settings. Save as Hardware/src/plot_fig2_hardware_traces.py.
Reads Hardware/hardware_dataset/*.csv, writes Hardware/results/figures/fig2_hardware_traces.png.
Run: python plot_fig2_hardware_traces.py   (optional arg: path to Hardware folder)"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Styling Configuration (IEEE Format)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "lines.linewidth": 1.2,
    "grid.alpha": 0.35,
    "grid.linestyle": "-"
})

# Path to the raw hardware dataset CSVs
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "hardware_dataset"

# 5 representative physical runs across health stages
# Using data_100.csv (not depressurized) to show steady 100% flow saturation
rep_files = [
    ("data_0.csv", "0.0% (Healthy)", "#2ca02c"),
    ("data_12_50.csv", "12.5% (Minor)", "#1f77b4"),
    ("data_37_5.csv", "37.5% (Moderate)", "#ff7f0e"),
    ("data_75.csv", "75.0% (Severe)", "#d62728"),
    ("data_100.csv", "100.0% (Severe, not depressurized)", "#9467bd"),
]

# ---------------------------------------------------------------------------
# 2. Create 2x2 Subplots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.5))

t_start, t_end = 2.0, 5.0
handles, labels = [], []

for fname, label, color in rep_files:
    file_path = DATA_DIR / fname
    if not file_path.exists():
        print(f"Warning: {file_path} not found.")
        continue

    df = pd.read_csv(file_path)
    t = df["Elapsed_Time_sec"].values
    p = df["Pressure_bar"].values
    q = df["Airflow_LPM"].values

    # Sub-window mask for 3-second zoom (2 to 5 s)
    mask_win = (t >= t_start) & (t <= t_end)
    t_win = t[mask_win]
    p_win = p[mask_win]
    q_win = q[mask_win]

    # Instantaneous Pneumatic Fluid Power [Watts]:
    # P = p_abs [bar] * 1e5 [Pa] * q [LPM] * (1e-3/60 m^3/s) = (p + 1.013) * q * (10/6)
    power_win = (p_win + 1.013) * q_win * (10.0 / 6.0)

    # (a) Line pressure, full 10 s run
    line, = axes[0, 0].plot(t, p, color=color, alpha=0.9, label=label)
    if label not in labels:
        handles.append(line)
        labels.append(label)

    # (b) Airflow, full 10 s run
    axes[1, 0].plot(t, q, color=color, alpha=0.9)

    # (c) Line pressure, 3 s window (2 to 5 s)
    axes[0, 1].plot(t_win, p_win, color=color, alpha=0.9)

    # (d) Fluid power, 3 s window (2 to 5 s)
    axes[1, 1].plot(t_win, power_win, color=color, alpha=0.9)

# ---------------------------------------------------------------------------
# 3. Titles, Axes Limits, and Thresholds
# ---------------------------------------------------------------------------

# Subplot (a): Line Pressure (Full 10 s)
axes[0, 0].set_title("(a) Line pressure, full 10 s run")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Line pressure (bar gauge)")
axes[0, 0].grid(True)

# Subplot (b): Airflow (Full 10 s)
line_lim = axes[1, 0].axhline(
    16.0,
    color="black",
    linestyle=":",
    linewidth=1.4,
    label="Sensor limit (16.0 LPM)"
)
axes[1, 0].set_title("(b) Airflow (37.5%, 75%, 100% stay at 16.0 LPM)")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Airflow (LPM)")
axes[1, 0].set_ylim(-1.0, 18.0)
axes[1, 0].grid(True)

# Append sensor limit line to legend
handles.append(line_lim)
labels.append("Sensor limit (16.0 LPM)")

# Subplot (c): Line Pressure (Zoomed 2 to 5 s)
axes[0, 1].set_title("(c) Line pressure, 3 s window (2 to 5 s)")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Line pressure (bar gauge)")
axes[0, 1].set_xlim(t_start, t_end)
axes[0, 1].grid(True)

# Subplot (d): Fluid Power (Zoomed 2 to 5 s)
axes[1, 1].set_title("(d) Fluid power from measured (saturated) flow")
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].set_ylabel("Fluid power (W)")
axes[1, 1].set_xlim(t_start, t_end)
axes[1, 1].grid(True)

# ---------------------------------------------------------------------------
# 4. Global Legend at Bottom
# ---------------------------------------------------------------------------
fig.legend(
    handles=handles,
    labels=labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
    frameon=False
)

# ---------------------------------------------------------------------------
# 5. Output
# ---------------------------------------------------------------------------
plt.tight_layout()
out_dir = BASE / "results" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "fig2_hardware_traces.png"
plt.savefig(out_file, bbox_inches="tight")
print(f"Saved figure to: {out_file}")