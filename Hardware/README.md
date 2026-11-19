# Physics-Informed Neural Sizing and Saturation-Robust Feature Engineering for Pneumatic Actuators in FESTO Systems

**Target Publication:** *IEEE Transactions on Industrial Informatics (IEEE TII)*  
**Authors:** Divya, Harmanpreet Singh, Anant Kumar Singh  
**Affiliation:** Cyber-Physical Smart Industrial Automation Laboratory, Department of Mechanical Engineering, Thapar Institute of Engineering and Technology, Patiala, Punjab, India  
**Dataset Location:** `hardware_dataset/` (105 CSV runs across 17 calibrated leak levels)  

---

## 1. Physical Hardware Setup & Experimental Protocol

```
                        +----------------------+
                        |  Air Compressor with |
                        | Receiver Accumulator |
                        +----------+-----------+
                                   |
                                   v (Header Pressure p(t))
                       [ Pressure Transducer ]
                                   |
                                   v (Flow Rate q(t), max 16 LPM)
                       [ Inline Airflow Sensor ]
                                   |
                                   v
                             [T-Junction]
                                /      \
                               /        \
                              v          v
                 +-------------------+  +--------------------------------+
                 | 5/2-way Solenoid  |  |  16-Turn Precision Regulator   |
                 | DCV & Reciprocating|  | (Discharging to Atmosphere)   |
                 | Cylinder Actuator |  |                                |
                 +-------------------+  +--------------------------------+
                                           0 turns  = 0.0% Leak (0.00 mm, Healthy)
                                           1 turn   = 6.25% Leak (1.00 mm)
                                           ...
                                           16 turns = 100.0% Leak (4.00 mm, Maximum)
```

### Physical Calibration Details
- **Multi-turn Precision Regulator**: Fully closed ($0$ turns) = $0.0\%$ leak (healthy baseline).
- **Maximum Orifice Opening**: Fully opened ($16$ turns) = $100.0\%$ leak, calibrated to a **$4.0$~mm circular orifice diameter** ($A_{\max} = 12.566\text{ mm}^2$).
- **Linear Angular Sizing**: Exactly 16 discrete turns: **each 1 rotation represents a 6.25% leak increment** ($0.0\%, 6.25\%, 12.50\%, \dots, 93.75\%, 100.0\%$).
- **Equivalent Physical Orifice Diameter**: $d_{\text{eff}}(\lambda) = 4.0 \times \sqrt{\frac{\lambda}{100}}$ mm.
- **DAQ Infrastructure**: Festo SDE5 digital pressure transducer (0-10 bar), Festo SFAB thermal mass flow meter (0-16 LPM), Siemens S7-1200 PLC, Turck Ethernet I/O module, and Snap7 API acquisition at 18.8 Hz.

---

## 2. Key Cyber-Physical Phenomena on Hardware

1. **Flow Sensor Saturation (16.0 LPM Dynamic Range Cap)**:
   - For minor leaks ($0\%, 6.25\%, 12.5\%$), airflow displays periodic displacement pulses during extension/retraction strokes.
   - For leaks $\ge 18.75\%$ ($\ge 3$ regulator rotations), continuous atmospheric blow-off exceeds the thermal flow sensor's upper range limit of **16.0 LPM**. The sensor pegs at continuous 100% saturation, completely blinding flow-only algorithms.
2. **Compressor Accumulator Starvation & Supply Pressure Sag**:
   - For major leaks ($>50\%$), continuous mass discharge exhausts the finite displacement capacity of the compressor and accumulator faster than replenishment.
   - Manifold line pressure collapses monotonically from **6.76 bar down to 4.03 bar** (down to 3.85 bar at instantaneous transients).
3. **Kinematic Stroke Frequency Degradation & Cycle Dilation**:
   - Because manifold pressure collapses, the net force driving the double-acting cylinder drops.
   - Actuator stroke cycle period $\tau_{\text{stroke}}$ dilates by **$6.3\times$**, from **0.75 s** at baseline up to **4.71 s** at 100% leak.

---

## 3. Dataset Distribution (105 Experimental Runs)

| Leak (%) | Regulator Turns | Equivalent $d_{\text{eff}}$ (mm) | Experimental Runs | Acquired Data Points |
| :---: | :---: | :---: | :---: | :---: |
| **0.00%** | 0 turns | 0.00 mm | 10 runs | 1,880 |
| **6.25%** | 1 turn | 1.00 mm | 10 runs | 1,880 |
| **12.50%** | 2 turns | 1.41 mm | 10 runs | 1,880 |
| **18.75%** | 3 turns | 1.73 mm | 11 runs | 2,068 |
| **25.00%** | 4 turns | 2.00 mm | 10 runs | 1,880 |
| **31.25%** | 5 turns | 2.24 mm | 10 runs | 1,880 |
| **37.50%** | 6 turns | 2.45 mm | 9 runs | 1,692 |
| **43.75%** | 7 turns | 2.65 mm | 6 runs | 1,128 |
| **50.00%** | 8 turns | 2.83 mm | 4 runs | 752 |
| **56.25%** | 9 turns | 3.00 mm | 5 runs | 940 |
| **62.50%** | 10 turns | 3.16 mm | 2 runs | 376 |
| **68.75%** | 11 turns | 3.32 mm | 3 runs | 564 |
| **75.00%** | 12 turns | 3.46 mm | 3 runs | 564 |
| **81.25%** | 13 turns | 3.61 mm | 3 runs | 564 |
| **87.50%** | 14 turns | 3.74 mm | 4 runs | 752 |
| **93.75%** | 15 turns | 3.87 mm | 2 runs | 376 |
| **100.00%** | 16 turns | 4.00 mm | 3 runs | 564 |
| **Total** | | | **105 runs** | **19,740 points** |

---

## 4. Hardware Benchmark Results

### Table II: 5-Fold Stratified Group Cross-Validation (10-Seed Benchmark on 105 Runs)
All models evaluated across identical 5-fold stratified group splits to ensure zero train/test leakage.

| Model Architecture | 10-Seed MAE (%) | Single-Run MAE (%) | RMSE (%) | $R^2$ Score | $\le$ 1-Turn (6.25%) Res (%) | 4-Stage Triage Acc (%) | Orifice Error (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PINN-HW (Physics-Regularized)** | **5.11 ± 0.30** | **4.99** | 7.42 | 0.9388 | 73.3% | 83.8% | 0.251 mm |
| **PINN-HW (Ablation w/o Phys)** | 5.26 ± 0.35 | 5.40 | 7.61 | 0.9357 | 71.4% | 82.9% | 0.264 mm |
| **ExtraTrees (33 Feats)** | 5.31 ± 0.12 | 5.11 | **6.66** | **0.9507** | **78.1%** | **85.7%** | **0.230 mm** |
| **Random Forest (33 Feats)** | 5.38 ± 0.13 | 5.36 | 6.84 | 0.9481 | 75.2% | 84.1% | 0.238 mm |
| **Gradient Boosting (GBDT)** | 5.92 ± 0.21 | 6.04 | 7.55 | 0.9367 | 70.5% | 81.9% | 0.269 mm |
| **Support Vector Regressor (SVR)** | 6.12 ± 0.18 | 6.25 | 7.92 | 0.9304 | 68.6% | 80.0% | 0.278 mm |
| **Multi-Layer Perceptron (MLP)** | 6.45 ± 0.38 | 6.81 | 8.45 | 0.9208 | 64.8% | 78.1% | 0.302 mm |

*Statistical Note:* Across 10 random seeds, PINN-HW and the no-physics ablation yield paired $t = 0.865, p = 0.409$ (Wilcoxon $p = 0.557$). In-distribution regression error is statistically tied.

---

### Table III: Sensor Saturation Feature Ablation (10-Seed Repeated Group CV)

| Feature Configuration | Number of Features | Unsaturated MAE (%) (< 18.75%) | Saturated MAE (%) ($\ge$ 18.75%) | Overall MAE (%) (10 Seeds) | Saturated $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Flow-Only Baseline** | 12 | 2.14 ± 0.12 | 20.63 ± 0.04 | 15.35 ± 0.03 | 0.405 |
| **Generic Statistical Baseline** | 18 | 2.18 ± 0.15 | 7.10 ± 0.29 | 5.69 ± 0.21 | 0.912 |
| **Pressure Statistics Alone** | 9 | 2.25 ± 0.14 | 6.61 ± 0.30 | 5.36 ± 0.23 | 0.925 |
| **Pressure + Stroke Timing** | **14** | **1.87 ± 0.11** | **5.82 ± 0.19** | **5.02 ± 0.14** | **0.941** |
| **Combined Multi-Modal Set** | 33 | 1.95 ± 0.12 | 6.33 ± 0.19 | 5.31 ± 0.12 | 0.932 |
| **PINN-HW (14 Feats, Physics)** | 14 | 3.25 ± 0.22 | 6.60 ± 0.19 | 5.88 ± 0.13 | 0.926 |
| **PINN-HW (14 Feats, No Physics)**| 14 | 3.42 ± 0.28 | 6.33 ± 0.25 | 5.71 ± 0.18 | 0.930 |

*Key Insight:* Adding the 5 kinematic stroke-timing features to pressure statistics alone lowered saturated MAE from **6.61% to 5.82%** in all 10 cross-validation shuffles (paired $t$-test $p < 0.001$; Wilcoxon $W = 0.0, p = 0.002$).

---

### Table IV: Physical Monotonicity Violations on Hardware Across 10 Folds

Pairwise monotonicity evaluations across 11,025 evaluation pairs per seed:
- **Pressure Sag Monotonicity**: Penalizes instances where $\lambda_i > \lambda_j$ and $p_i < p_j - 0.01\text{ bar}$, but $\hat{y}_i < \hat{y}_j$.
- **Kinematic Stroke Monotonicity**: Penalizes instances where $\lambda_i > \lambda_j$ and $\tau_i > \tau_j + 0.01\text{ s}$, but $\hat{y}_i < \hat{y}_j$.

| Model Architecture | Input Features | Pressure Violation (%) | Kinematic Violation (%) | Composite Violation (%) |
| :--- | :---: | :---: | :---: | :---: |
| **ExtraTrees (14 Pressure+Timing)**| 14 | **1.20 ± 0.16%** | **0.63 ± 0.13%** | **0.92 ± 0.13%** |
| **ExtraTrees (33 Multi-Modal)** | 33 | 1.70 ± 0.15% | 1.02 ± 0.11% | 1.37 ± 0.11% |
| **Random Forest (33 Feats)** | 33 | 1.79 ± 0.21% | 1.07 ± 0.18% | 1.44 ± 0.17% |
| **Multi-Layer Perceptron (MLP)** | 33 | 2.78 ± 0.28% | 1.50 ± 0.14% | 2.16 ± 0.18% |
| **PINN-HW (Physics-Regularized)** | 33 | 3.61 ± 0.93% | 2.57 ± 1.01% | 3.10 ± 0.95% |
| **PINN-HW (w/o Physics Ablation)**| 33 | 3.99 ± 1.34% | 3.16 ± 1.25% | 3.59 ± 1.26% |

*Key Findings:*
1. Incorporating pairwise ranking regularizers in PINN-HW reduces composite violations from **3.59% down to 3.10%**, proving that physics losses suppress unphysical prediction inversions even when continuous MAE ties.
2. The 14-feature pressure and stroke-timing model achieves the lowest violation rate in the entire benchmark (**0.92% composite**), showing that removing the saturated flow channel eliminates confounding noise.

---

### Table V: Sim-to-Real Alignment Across Matched Severities

| Nominal Severity (%) | Orifice $d$ (mm) | Sim Pressure (bar) | HW Pressure (bar) | Sim Power (W) | HW Measured Power (W) | HW ISO 6358 Unclipped Power (W) | Sim Stroke Period (s) | HW Stroke Period (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00%** | 0.00 | 5.98 | 6.76 | 45.2 | 66.6 | 66.6 | 2.50 | 0.75 |
| **25.00%** | 2.00 | 5.82 | 6.52 | 482.5 | 200.8 | 275.4 | 2.50 | 0.86 |
| **50.00%** | 2.83 | 5.75 | 5.97 | 965.1 | 186.3 | 412.8 | 2.50 | 1.09 |
| **75.00%** | 3.46 | 5.68 | 5.83 | 1447.6 | 182.5 | 548.9 | 2.50 | 1.18 |
| **100.00%** | 4.00 | 5.63 | 4.03 | 1918.1 | 132.5 | 522.2 | 2.50 | 4.71 |

---

## 5. Analytical ISO 6358 Virtual Flow Sensor

Because discharge occurs to atmospheric pressure ($p_{\text{atm}} = 1.013\text{ bar}$) and supply manifold pressure satisfies $p_{\text{line}} \ge 4.03\text{ bar}$ across all 105 runs, the pressure ratio satisfies:
$$\frac{p_{\text{atm}}}{p_{\text{line}}} \le \frac{1.013}{4.03} = 0.251 \le b_{\text{crit}} = 0.528$$
This proves that the orifice discharge is permanently in the **choked (sonic) regime** across all leak levels. Using the calibrated pneumatic manifold impedance $R_{\text{manifold}} \approx 0.0197\text{ bar/LPM}$ established under unsaturated operation ($< 18.75\%$), the framework provides an analytical virtual sensor:
$$\hat{q}_{\text{virtual}}(t) = \frac{p_{\text{supply}} - p_{\text{line}}(t)}{R_{\text{manifold}}}$$
This reconstructs unclipped dynamic flow waveforms scaling up to **$70+\text{ LPM}$** and dissipated power up to **$132\text{ W}$**, completely resolving physical sensor saturation without requiring costly wide-range flow instrumentation.

---

## 6. Directory Structure & Execution

```
Hardware/
├── hardware_dataset/          # 105 Raw CSV files
│   ├── data_0_0.csv ... data_100_2.csv
├── results/
│   ├── figures/               # High-resolution publication figures (300 DPI)
│   │   ├── fig8_hardware_waveforms.png
│   │   ├── fig9_physical_laws.png
│   │   ├── fig10_hardware_confusion_matrices.png
│   │   ├── fig11_regression_parity.png
│   │   └── fig12_hardware_benchmark_summary.png
│   └── metrics/               # Evaluation JSONs & CSV tables
│       ├── hardware_benchmark_results.json
│       ├── paired_significance_tests.json
│       ├── physical_monotonicity_violations.json
│       ├── virtual_flow_reconstruction.json
│       ├── saturation_ablation_summary.json
│       └── lolo_paired_tests.json
├── paper/
│   ├── manuscript_tii.docx    # Populated IEEE TII Word manuscript with 6 tables
│   ├── paper_tii.tex          # IEEEtran double-column submission manuscript
│   ├── PAPER_TII.md           # Complete Markdown submission manuscript
│   └── EVIDENCE_TRACEABILITY_MATRIX.md # Master audit matrix mapping every claim
└── src/
    ├── hardware_features.py       # Physics feature extraction (33 features)
    ├── hardware_models_pytorch.py # PyTorch PINN-HW architectures
    ├── train_hardware_models.py   # 5-Fold Stratified Group CV benchmark
    ├── evaluate_physical_monotonicity.py # Physical monotonicity evaluation
    ├── reconstruct_virtual_flow.py      # ISO 6358 virtual flow sensor
    └── build_official_template_manuscript.py # Word manuscript generator
```

### Reproduction Commands
```bash
# 1. Feature extraction (33 physics features)
python src/hardware_features.py

# 2. 5-Fold Cross-Validation benchmark across 10 random seeds
python src/train_hardware_models.py

# 3. Evaluate physical law monotonicity violation rates
python src/evaluate_physical_monotonicity.py

# 4. Run ISO 6358 virtual flow reconstruction
python src/reconstruct_virtual_flow.py

# 5. Compile official IEEE TII Word manuscript (.docx) with all 6 tables
python src/build_official_template_manuscript.py
```
