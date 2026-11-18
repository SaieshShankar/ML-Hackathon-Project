# PathSnap champion

**Phase 2 default · 0.9127 student-test Macro F1 · 91.98% Top-1 · CPU only**

The tuned model uses 51 LightGBM features and topology-aware Viterbi with learned transition scores. The classifier was fitted on all 80 training trips. The separate transition scorer used 78 trips after applying the 50 m alignment-quality filter. Phase 2 reached 0.8825 Macro F1 across the original five training folds. The untuned model reached 0.9065 student-test Macro F1 and 92.84% Top-1. Both sets of weights are available. See the [Phase 2 results](results/phase2/REPORT.md).

## Start here

| File | What it does |
|---|---|
| `prepare.py` | Turns the original student ZIP into candidates, 51 features and derived labels. |
| `train.py` | Retrains the classifier on all 80 trips and fits the learned transition scorer. |
| `tune.py` | Runs Bayesian tuning, refits both versions and evaluates the same whole-trip partitions. |
| `serve_api.py` | Starts the FastAPI service with the frozen Phase 2 model on Laptop 1. |
| `evaluator_ui/` | Browser UI, offline map assets and the Laptop 2 launcher. |
| `predict.py` | Predicts the arc for each observation in one trip. Uses GPS, nodes and arcs. |
| `evaluate.py` | Scores the bundled or retrained model on the prepared student-test trips. |
| `pathsnap/` | Candidate generation, training-label alignment, heading, context, graph, Viterbi and metrics. |
| `model/` | Phase 1 weights at the root and the Phase 2 default under `phase2-tuned/`. |
| `results/` | [Experiment report](results/REPORT.md), comparison and per-trip scores. |
| `Old Models/` | Earlier benchmark, original experiments, demos and working archives. |

## Reproduce the 10-trip student test

Run from this directory with Python 3.11. Supply your original copy of Student folder.zip.

```sh
python -m pip install -r requirements.txt
python prepare.py --dataset "Student folder.zip" --split student_test
python evaluate.py
```

Use `python evaluate.py --legacy` to reproduce the original 0.9065 result. The reserved evaluator trips were outside both experiments.

## Predict a trip

```sh
python predict.py --track trip.track --nodes trip.nodes --arcs trip.arcs --output predictions.csv
```

The prediction path uses no reference route. Arc IDs are zero-based row indices in the trip's arc file. Arc ID -1 means unmatched. The output marks starts, missing candidates, resumes, GPS gaps and bounded disconnected transitions so a map can avoid drawing a road link across those breaks. Use `--model model` for the untuned weights and the Phase 2 default decoder settings.

## Refit the original baseline on CPU

```sh
python prepare.py --dataset "Student folder.zip"
python train.py --output trained-model
python evaluate.py --model trained-model --legacy
```

Retraining writes to trained-model by default, leaving the verified weights available. The 10 student-test trips are read by the evaluation command. The original configuration search is under [Old Models](Old%20Models/Integrated%20experiment%20sweep/source/README.md).

## Run Phase 2 on CPU

```sh
python prepare.py --dataset "Student folder.zip" --split train
python tune.py --dataset "Student folder.zip" --prepared prepared --output phase2-work/run
python verify_decoder.py
```

The optimiser uses a Gaussian process with expected improvement. It first tunes the LightGBM scorer against decoded F1 on five original training folds, then tunes the decoder using the selected scorer's out-of-fold predictions. It freezes the settings before scoring any student-test trip and saves separate native model files for the untuned and tuned variants.

## Two-laptop demo

Follow the [two-laptop guide](docs/TWO_LAPTOP_DEMO.md) for setup, rehearsal and the switch to the organiser's evaluation data. Laptop 1 runs inference and scoring. Laptop 2 gets the evaluator UI package and Laptop 1's LAN URL.
