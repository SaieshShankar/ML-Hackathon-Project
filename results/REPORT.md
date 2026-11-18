# PathSnap integrated LightGBM experiment

The pipeline selected on the training folds was **LightGBM + context + heading, all trips** with the `topology_learned_no_reset` decoder. It scored **0.8806** Macro F1 across 80 validation trips, then **0.9065 Macro F1** and **92.84% Top-1** on the 10 student-test trips.

The best observed student-test result among fresh variants was **0.9065**. The selected pipeline was +2.73 percentage points relative to the attached LightGBM replay. 6 of its 10 trips reached 0.90 Macro F1.

| Pipeline | Decoder | CV Macro F1 | Student-test Macro F1 | Student-test Top-1 |
|---|---|---:|---:|---:|
| Previous in-house best (XGBoost Ranker + direction rules) | `direction rules` | 0.8460 | 0.8647 | 90.37% |
| Attached LightGBM (replayed) | `legacy` | 0.8664 | 0.8792 | 92.18% |
| LightGBM + context + heading, all trips | `topology_learned_no_reset` | 0.8806 | 0.9065 | 92.84% |
| LightGBM + context, filtered | `topology_learned_no_reset` | 0.8805 | 0.9019 | 93.50% |
| LightGBM + context + heading, filtered | `topology_learned_no_reset` | 0.8798 | 0.9035 | 93.27% |

## Method

- All variants ran on CPU, with the same trip splits, eight candidate arcs and existing aligned reference labels.
- The 33-feature baseline combines the original 10 features with adjacent observations' candidate and road-connectivity context. The 51-feature variants add 18 multiscale direction features.
- The filter omits trips `00000033` and `00000054` from fitting where enabled. All 80 training trips remain in the CV scoring population.
- Decoder settings were selected on the training folds before student-test scoring. The grid tested the legacy topology, different emission weights, explicit reverse costs, direction-sensitive penalties, gap resets, graph distance and learned transition potentials.
- The attached CV value is the reported all-trip figure. Its student-test value comes from the independent model replay. The student-test split has been used for development comparisons. Reserved evaluator trips were outside the experiment.

[Full summary metrics](comparison.csv) · [Selected model by trip](selected_per_trip.csv) · [Comparison chart](macro_f1_comparison.png)
