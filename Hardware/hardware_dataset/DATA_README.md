# Hardware dataset

105 runs from the Festo MPS.pa rig, one CSV per run, 10 s each, 19,648 samples in total.

## File names
`data_<level>[_<run>].csv`. The leak level is written with `_` instead of the decimal point, and the first run of a level has no run suffix.

| File | Leak level | Run |
|---|---|---|
| `data_0.csv`, `data_0_1.csv` … `data_0_9.csv` | 0% | 1st, 2nd … 10th |
| `data_6_25.csv` … | 6.25% | |
| `data_12_50.csv` … | 12.5% | |
| `data_100.csv`, `data_100_1.csv`, `data_100_2.csv` | 100% | 1st, 2nd, 3rd |

The 17 levels run from 0% to 100% in 6.25% steps (0 to 16 regulator turns). Runs per level: 10 for 0, 6.25, 12.5, 25 and 31.25%; 11 for 18.75%; 9 for 37.5%; 6 for 43.75%; 4 for 50 and 87.5%; 5 for 56.25%; 3 for 68.75, 75, 81.25 and 100%; 2 for 62.5 and 93.75%.

## Columns
| Column | Unit | Meaning |
|---|---|---|
| `Elapsed_Time_sec` | s | Time since the start of the run (0 to about 10 s) |
| `Clock_Time` | hh:mm:ss.mmm | Wall-clock time of the sample |
| `Pressure_bar` | bar | Line pressure |
| `Airflow_LPM` | L/min | Airflow. The sensor reads at most 16.0 LPM. |

Sampling is about 18.8 Hz, with 186 to 188 samples per run.

## Notes
- Flow saturation: for leaks of 18.75% and above, `Airflow_LPM` sits at 16.0 for much of the run, so 75 of the 105 runs are saturated.
- `data_100_2.csv` is a depressurized run: `Pressure_bar` is 0.0 throughout. It is kept in all results.
- Each file name is also the group ID in cross-validation, so all samples of a run stay in one fold.
- Raw data: no filtering or resampling was applied.