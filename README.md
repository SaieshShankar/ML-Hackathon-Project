# PathSnap

PathSnap matches noisy GPS observations to candidate road arcs. This repository contains the five-model, classifier-only benchmark and the offline Folium demo for a randomly selected student-test trip.

## Results

| Model | CV Macro F1 | Student-test Macro F1 | Student-test Top-1 |
|---|---:|---:|---:|
| Histogram Gradient Boosting | **0.6752** | **0.6903** | **83.74%** |
| Random Forest | 0.6691 | 0.6788 | 83.45% |
| Shallow MLP | 0.6611 | 0.6804 | 83.35% |
| Logistic Regression | 0.6362 | 0.6525 | 81.94% |
| Shallow Decision Tree | 0.6120 | 0.6181 | 80.91% |
| Nearest-segment heuristic | 0.3534 | 0.3492 | 53.86% |

See [comparison metrics](results/comparison.csv), [fold metrics](results/cv_fold_metrics.csv), and the [review walkthrough](DEMO.md).

## Reproduce the benchmark

Use Python 3.11 and your original student data ZIP. The archive contains 80 training trips and 10 student-test trips. Each trip contains a GPS track, road nodes, directed arcs, and an ordered reference route.

```sh
python -m venv .venv
# Activate .venv for your shell, then run:
python -m pip install -r requirements.txt
python benchmark.py prepare --dataset "Student folder.zip" --cache data.joblib --output run
python benchmark.py run --cache data.joblib --output run
python summarize.py run
python verify_alignment.py
```

The run directory contains the dataset audit, digest, model settings, fold assignments, saved predictions, metrics, and five fitted estimators. The summariser checks the recorded metrics against the predictions and generates the comparison chart. The alignment check compares the route alignment against a brute-force transition calculation on synthetic cases.

## Build the Folium demo

After running the benchmark:

```sh
python fetch_demo_assets.py --output leaflet
python demo_map.py --dataset "Student folder.zip" --evidence run --vendor leaflet --trip 00000076 --output PathSnap_Folium_Demo.html
```

The asset step downloads and verifies pinned Leaflet and jQuery files. Open the generated HTML in a browser. The demo works offline and includes toggleable layers, a corrected match, and a remaining error. Trip 00000076 was drawn from the student-test split before inspecting scores. Boosting reached 82.77% Top-1 on that trip, and nearest segment reached 41.12%.

## Evaluation design

- Shortlist 64 nearby arc midpoints and keep the eight nearest arcs by point-to-segment distance. Score candidate pairs using distance, heading, movement, and segment geometry. The candidate shortlist is approximate.
- Align the reference route monotonically to timestamped GPS observations using point-to-segment distance and a 2 m penalty for each skipped intermediate route position. This produces proxy labels for fitting and scoring. Raw GPS observations were kept intact.
- Fit and validate with five whole-trip folds. Each fit uses alternating observations from every fitting trip. Every held-out observation is scored. Final refits use the 80 training trips and are evaluated on all 10 student-test trips.
- Report Macro F1 per trip and average the trip scores. Top-1 counts exact aligned-arc matches and includes candidate misses as errors. Candidate log loss uses binary candidate scores. Direct continuity counts adjacent picks that share an arc or connect end-node to start-node.
- The boosted model uses 75 iterations, 31 maximum leaves per tree, and a 0.10 learning rate. Full configurations are in the benchmark source and are exported with each run.

The classifier results precede Bayesian tuning and HMM/Viterbi reconstruction. The reserved evaluator trips and the live two-laptop integration are outside this baseline.
