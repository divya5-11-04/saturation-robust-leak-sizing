# Simulation study (Festo FluidSIM)

Simulated leak traces and the PI-DTNet model used in the simulation part of the paper (Table VI).

## Data
- A reciprocating cylinder circuit in Festo FluidSIM, with 7 leak settings: 0, 1, 5, 25, 50, 75 and 100%.
- Each trace is 10 s at 100 Hz (1,001 points), containing 4 actuation cycles of 2.5 s.
- Traces were digitized from FluidSIM plot captures (`src/digitize_curves.py`, source images `0% Leak.png` to `100% Leak.png` in this folder). The digitized traces are in `data/raw_extracted/`.
- Splits are by cycle: cycles 0 and 1 for training (56 samples), cycle 2 for validation (28), cycle 3 for testing (28 = 7 severities x 4 noise levels: clean, 30, 20, 15 dB). An extra out-of-distribution test set uses 10 dB noise.

## PI-DTNet
A 1-D CNN (32, 64, 64 filters) followed by a 2-layer bidirectional LSTM (32 hidden units per direction, dropout 0.2), attention pooling, and a classification head and a regression head. The loss is classification + 0.5 x regression + 0.1 x mass-flow term (ISO 6358) + 0.05 x fluid-power monotonicity term. AdamW, learning rate 1e-3, weight decay 1e-4, cosine schedule to 1e-5, batch size 32, 25 epochs. 85,481 parameters.

## Results (held-out cycle, n = 28)
MAE is mean ± sample SD over 10 initializations (seeds 42, 10, 20, 30, 40, 50, 60, 70, 80, 90) for the first three rows, and a single run for the others. Accuracy and ROC-AUC are single-run values.

| Model | Acc. (%) | ROC-AUC | MAE (%) | OOD MAE, 10 dB (%) |
|:--|:-:|:-:|:-:|:-:|
| PI-DTNet | 42.86 | 0.8765 | 9.81 ± 2.27 | 10.28 ± 2.24 |
| No physics loss | 42.86 | 0.9762 | 12.52 ± 3.04 | 12.86 ± 3.06 |
| No physics features | 42.86 | 0.8810 | 12.67 ± 2.53 | - |
| 1D-CNN | 71.43 | 0.9167 | 7.14 | - |
| LSTM | 57.14 | 0.9048 | 6.72 | - |
| Random forest | 85.71 | 1.0000 | 1.24 | - |
| SVM | 60.71 | 0.9479 | 17.15 | - |

Physics loss vs none (clean test set): 12.52% → 9.81% (2.71 pp), paired t = 7.334, p = 4.4e-5, Wilcoxon p = 0.002, lower in 10 of 10 seeds. At 10 dB: 12.86% → 10.28%, t = 6.770, p = 8.2e-5.

These results come from one fixed test set. They show consistency across initializations, not generalization.

## Layout
```
Simulation/
  0% Leak.png ... 100% Leak.png    FluidSIM captures (input to digitization)
  src/
    digitize_curves.py             digitize captures into data/raw_extracted/
    fluid_physics.py               ISO 6358 flow and thermodynamics helpers
    dataset_generator.py           features, noise, cycle-based splits
    physics_aware_models.py        PI-DTNet and physics losses
    train_and_evaluate.py          training and evaluation of all models
    run_10seed_sim_benchmark.py    10-seed physics ablation
  data/raw_extracted/  data/processed/
  results/{figures,metrics,models,digitization_overlays}/
```

## Reproduce
Run from the `Simulation/` folder.
```bash
python src/digitize_curves.py
python src/dataset_generator.py
python src/train_and_evaluate.py
python src/run_10seed_sim_benchmark.py     # writes results/metrics/sim_10seed_physics_ablation.json
```
Neural results can differ slightly between machines.