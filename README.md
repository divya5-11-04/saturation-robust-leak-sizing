# Saturation-Robust Leak Sizing in Pneumatic Actuators

Code, data and results for the paper **"Physics-Aware Machine Learning for Saturation-Robust Leak Sizing in Pneumatic Actuators"**.

A leak in a pneumatic actuator is sized from its pressure signal and stroke timing. This works even when the flow sensor is saturated (it reads at most 16.0 LPM, and 75 of our 105 runs sit at that limit). We compare tree models, a physics-regularized network (PINN-HW) and feature sets on a Festo MPS.pa rig, and test them with 10-seed grouped cross-validation and leave-one-level-out evaluation.

## Contents
| Folder | What it holds |
|---|---|
| `Hardware/` | 105 measured runs, feature extraction, benchmark scripts, result tables and figures |
| `Simulation/` | Simulated leak data and the PI-DTNet model used for the simulation study |

Each folder has its own README with details.

## Main findings (hardware, 105 runs)
- A 14-feature ExtraTrees model (9 pressure statistics + 5 stroke-timing features) gives the lowest error: 5.02 ± 0.14% MAE, against 5.31 ± 0.12% for 33 features and 15.74 ± 0.05% for flow only.
- On saturated runs (leak >= 18.75%), flow-only features give 20.63% MAE. Pressure plus stroke timing gives 5.82%.
- The physics loss in PINN-HW does not give a significant gain over the same network without it (5.38 vs 5.26% MAE, p = 0.40 over 10 seeds; composite monotonicity violations 3.66 vs 3.19%, Nadeau-Bengio p = 0.22).
- Leave-one-level-out (15 unseen levels): PINN-HW 7.01%, ExtraTrees-14 7.86%, ExtraTrees-33 8.57% mean MAE. These differences are not significant (p between 0.09 and 0.34).

Full tables are in `Hardware/README.md`.

## Paper to code
Table numbers follow the paper. The scripts label them one number higher.

| Paper item | Script | Output |
|---|---|---|
| Table I (benchmark) | `Hardware/src/run_authoritative_benchmarks.py` | `Hardware/results/metrics/table_II_authoritative.csv` |
| Table II (saturation ablation) | `Hardware/src/run_saturation_ablation_benchmarks.py` | `table_III_authoritative.csv`, `saturation_ablation_summary.json` |
| Table III (monotonicity violations) | `Hardware/src/run_authoritative_benchmarks.py`, `evaluate_physical_monotonicity.py` | `table_IV_authoritative.csv` |
| Leave-one-level-out results | `Hardware/src/run_authoritative_benchmarks.py` | `table_lolo_summary_authoritative.csv`, `lolo_paired_tests_authoritative.json` |
| Nested feature selection | `Hardware/src/run_nested_feature_selection.py` | printed to console |
| Margin sensitivity | `Hardware/src/run_margin_sensitivity.py` | `margin_sensitivity.csv` |
| Flow reconstruction | `Hardware/src/reconstruct_virtual_flow.py` | `virtual_flow_reconstruction.json` |
| Simulation vs hardware table | `Hardware/src/sim_to_real.py` | `sim_to_real_comparison.csv` |
| Figs. 2 to 4 | `plot_fig2_hardware_traces.py`, `plot_fig3_measured_behavior.py`, `plot_fig4_saturation_ablation.py` | `Hardware/results/figures/` |
| Simulation study | `Simulation/src/` | `Simulation/results/` |

## Quick start
Python 3.12.
```bash
git clone https://github.com/divya5-11-04/saturation-robust-leak-sizing
cd saturation-robust-leak-sizing
pip install -r requirements.txt
cd Hardware
python src/run_authoritative_benchmarks.py     # about 14 min on CPU
```
Run the hardware scripts from `Hardware/`. See `Hardware/README.md` for every command.

## Reproducibility
- Data splits: `StratifiedGroupKFold` with the run file as the group, so no run is split across train and test.
- Seeds: 42, 100, 2024, 7, 999, 123, 777, 314, 888, 555.
- Feature scaling is fitted on training folds only.
- Tree models reproduce exactly. PINN numbers can differ slightly between machines (PyTorch version and hardware).
- Package versions are pinned in `requirements.txt`.

## Data
`Hardware/hardware_dataset/` holds 105 runs of pressure and flow signals. See `Hardware/hardware_dataset/DATA_README.md` for the columns and file naming.

## Citation
If you use this repository, please cite the paper. See `CITATION.cff`. An archived release is on Zenodo: [DOI 10.5281/zenodo.24368156](https://doi.org/10.5281/zenodo.24368156).

## License
Code: MIT (see `LICENSE`). Data in `Hardware/hardware_dataset/`: CC-BY 4.0.