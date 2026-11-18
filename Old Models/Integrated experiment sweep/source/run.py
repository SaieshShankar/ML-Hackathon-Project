"""Compare LightGBM context, heading, trip filtering and topology decoders on CPU."""
import argparse
from dataclasses import asdict
from hashlib import file_digest
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from zipfile import ZipFile

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from baseline_viterbi import csv_read, csv_write, evaluate
from methods import DECODERS, ROUTE_COSTS, context_features, decode, learned_potentials, topology_classes

SEED = 352
THREADS = 4
FILTER_METRES = 50.
LGBM_PARAMS = dict(
    n_estimators=250, learning_rate=.06, num_leaves=63, min_child_samples=40,
    subsample=.8, subsample_freq=1, colsample_bytree=.8, reg_lambda=1.,
    device_type="cpu", n_jobs=THREADS, verbose=-1, random_state=SEED,
    deterministic=True, force_col_wise=True,
)
VARIANTS = {
    "context_clean": (33, True),
    "context_heading_clean": (51, True),
    "context_heading_all": (51, False),
}
FIELDS = ("original", "more", "prior", "candidates", "reference", "arcs", "route",
          "finite", "mismatch", "reset", "transition")


def load(path, keys=FIELDS):
    with np.load(path) as data:
        return {k: data[k] for k in keys}


def make_features(raw):
    context = context_features(raw["original"], raw["candidates"], raw["arcs"])
    return np.concatenate((context, raw["more"]), axis=-1)


def transition_examples(raw, labels, rng):
    target = np.where(labels.any(axis=1), labels.argmax(axis=1), -1)
    valid = np.flatnonzero((target[:-1] >= 0) & (target[1:] >= 0) & ~raw["reset"])
    previous, current = target[valid], target[valid + 1]
    k = labels.shape[1]
    negatives_prev = rng.integers(k, size=(len(valid), 8))
    negatives_curr = rng.integers(k, size=(len(valid), 8))
    negatives_prev[:, :3] = previous[:, None]
    negatives_curr[:, 3:6] = current[:, None]
    distinct = (negatives_prev != previous[:, None]) | (negatives_curr != current[:, None])
    transitions = raw["transition"]
    positives = transitions[valid, previous, current]
    negatives = transitions[valid[:, None], negatives_prev, negatives_curr][distinct]
    return (np.concatenate((positives, negatives)),
            np.r_[np.ones(len(positives), dtype=bool), np.zeros(len(negatives), dtype=bool)])


def prepare_training(paths):
    rng = np.random.default_rng(SEED)
    blocks = {}
    for trip, path in sorted(paths.items()):
        raw = load(path)
        labels = raw["candidates"] == raw["reference"][:, None]
        blocks[trip] = (make_features(raw), labels, *transition_examples(raw, labels, rng))
    return blocks


def fit_models(blocks, ids, rejected, lgb):
    models = {}
    timings = []
    for name, (count, clean) in VARIANTS.items():
        began = perf_counter()
        fit = [t for t in ids if not clean or t not in rejected]
        x = np.concatenate([blocks[t][0][..., :count].reshape(-1, count) for t in fit])
        y = np.concatenate([blocks[t][1].reshape(-1) for t in fit])
        model = lgb.LGBMClassifier(**LGBM_PARAMS).fit(x, y)
        models[name] = model.booster_
        elapsed = perf_counter()-began
        timings.append(dict(model=name, trips=len(fit), rows=len(y), seconds=elapsed))
        print(f"  {name}: fit on {len(fit)} trips, {len(y):,} rows, {elapsed:.1f}s", flush=True)
    began = perf_counter()
    retained = [t for t in ids if t not in rejected]
    features = np.concatenate([blocks[t][2] for t in retained])
    labels = np.concatenate([blocks[t][3] for t in retained])
    transitions = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=150, class_weight="balanced"))
    transitions.fit(features, labels)
    timings.append(dict(model="transition_logistic", trips=len(retained),
                        rows=len(labels), seconds=perf_counter()-began))
    return models, transitions, timings


def all_predictions(raw, models, transition_model, choices):
    features = make_features(raw)
    candidates = raw["candidates"]
    category, reverse = topology_classes(candidates, raw["arcs"])
    learned = learned_potentials(transition_model, raw["transition"])
    for name, configs in choices.items():
        count, _ = VARIANTS[name]
        flat = features[..., :count].reshape(-1, count)
        probabilities = models[name].predict(flat, num_threads=THREADS).reshape(candidates.shape)
        emit = np.log(np.clip(probabilities, 1e-6, 1.))
        yield name, "raw", emit.argmax(axis=1)
        for config in configs:
            yield name, config, decode(emit, category, reverse, raw, learned, DECODERS[config])


def select_decoders(rows, output):
    scores = {}
    for row in rows:
        scores.setdefault((row["model"], row["decoder"]), []).append(row)
    table = []
    for (model, decoder), group in scores.items():
        clean = [r["macro_f1"] for r in group if not r["filtered_from_fitting"]]
        all_scores = [r["macro_f1"] for r in group]
        assert len(all_scores) == 80 and len(clean) == 78
        table.append(dict(model=model, decoder=decoder, macro_f1_all=float(np.mean(all_scores)),
                          macro_f1_clean_diagnostic=float(np.mean(clean)),
                          **{f"fold_{i}": float(np.mean([r["macro_f1"] for r in group if r["fold"] == i]))
                             for i in range(1, 6)}))
    table.sort(key=lambda r: (-r["macro_f1_all"], r["model"], r["decoder"]))
    csv_write(output / "cv_search.csv", table)
    choices = {}
    for row in table:
        choices.setdefault(row["model"], row["decoder"])
    return choices, table[0]


def test_metrics(raw, selected_idx):
    predicted, stats = evaluate(selected_idx, raw["candidates"], raw["reference"], raw["route"], raw["arcs"],
                                raw["finite"], raw["reset"])
    wrong = predicted != raw["reference"]
    edges = raw["arcs"]
    reverse = wrong & np.all(edges[predicted] == edges[raw["reference"]][:, ::-1], axis=1)
    stats["reverse_errors"] = int(reverse.sum())
    return predicted, stats


def summarize_test(rows):
    table = []
    keys = {(r["model"], r["decoder"]) for r in rows}
    for model, decoder in sorted(keys):
        group = [r for r in rows if (r["model"], r["decoder"]) == (model, decoder)]
        assert len(group) == 10
        table.append(dict(model=model, decoder=decoder, trips=len(group),
                          macro_f1=float(np.mean([r["macro_f1"] for r in group])),
                          top1=sum(r["correct"] for r in group)/sum(r["observations"] for r in group),
                          direct_continuity=float(np.mean([r["direct_continuity"] for r in group])),
                          sequence_ratio=float(np.mean([r["reference_route_sequence_ratio"] for r in group])),
                          reverse_errors=sum(r["reverse_errors"] for r in group),
                          trips_at_090=sum(r["macro_f1"] >= .90 for r in group)))
    return sorted(table, key=lambda r: -r["macro_f1"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "set1", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--lightgbm-path", type=Path, help="Optional directory with a local LightGBM install.")
    args = parser.parse_args()
    if args.lightgbm_path:
        sys.path.append(str(args.lightgbm_path.resolve()))
    import lightgbm as lgb
    args.output.mkdir(exist_ok=True, parents=True)
    assignment = csv_read(args.set1 / "fold_assignments.csv")
    audit = csv_read(args.set1 / "dataset_audit.csv")
    train = sorted({r["trip_id"] for r in assignment})
    excluded = {r["trip_id"] for r in audit if r["split"] == "train"
                and float(r["median_aligned_distance_m"]) > FILTER_METRES}
    test = sorted({r["trip_id"] for r in audit if r["split"] == "student_test"})
    assert len(train) == 80 and len(test) == 10 and set(train).isdisjoint(test)
    with (args.set1 / "fold_assignments.csv").open("rb") as handle:
        fold_digest = file_digest(handle, "sha256").hexdigest()
    paths = {t: args.prepared / f"{t}.npz" for t in train + test}
    config = dict(device="cpu", threads=THREADS, python=platform.python_version(), numpy=np.__version__,
                  scipy=scipy.__version__, sklearn=sklearn.__version__, lightgbm=lgb.__version__,
                  fold_assignments_sha256=fold_digest,
                  dataset_sha256=(args.set1 / "dataset_sha256.txt").read_text().strip(),
                  training_filter_m=FILTER_METRES, excluded=sorted(excluded),
                  models=VARIANTS, lightgbm_params=LGBM_PARAMS,
                  topology_costs=ROUTE_COSTS.tolist(),
                  decoders={name: asdict(cfg) for name, cfg in DECODERS.items()})
    (args.output / "experiment.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    fit_log, cv_rows = [], []
    print(f"CPU run; filtered variants omit {sorted(excluded)} at fit time only.", flush=True)
    with threadpool_limits(limits=THREADS):
        blocks = prepare_training({t: paths[t] for t in train})
        all_choices = {name: list(DECODERS) for name in VARIANTS}
        for fold in range(1, 6):
            fit = sorted(r["trip_id"] for r in assignment if int(r["fold"]) == fold and r["role"] == "fit")
            validation = sorted(r["trip_id"] for r in assignment
                                if int(r["fold"]) == fold and r["role"] == "validation")
            assert len(fit) == 64 and len(validation) == 16 and set(fit).isdisjoint(validation)
            print(f"Fold {fold}/5", flush=True)
            models, transition, timing = fit_models(blocks, fit, excluded, lgb)
            fit_log.extend(dict(fold=fold, **r) for r in timing)
            for trip in validation:
                raw = load(paths[trip])
                for model, decoder, ix in all_predictions(raw, models, transition, all_choices):
                    predicted = raw["candidates"][np.arange(len(ix)), ix]
                    f1 = f1_score(raw["reference"], predicted, average="macro", zero_division=0)
                    cv_rows.append(dict(fold=fold, trip_id=trip, model=model, decoder=decoder,
                                        filtered_from_fitting=trip in excluded, macro_f1=f1))
            csv_write(args.output / "cv_per_trip.csv", cv_rows)
            csv_write(args.output / "fit_timings.csv", fit_log)
            print(f"Fold {fold}/5 complete", flush=True)

        choices, winner = select_decoders(cv_rows, args.output)
        config.update(selected_by_cv=winner, decoder_choice_per_model=choices)
        (args.output / "experiment.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"Frozen global winner: {winner['model']}/{winner['decoder']}, "
              f"CV Macro F1 {winner['macro_f1_all']:.6f}", flush=True)

        models, transition, timing = fit_models(blocks, train, excluded, lgb)
        fit_log.extend(dict(fold=0, **r) for r in timing)
        csv_write(args.output / "fit_timings.csv", fit_log)
        saved = args.output / "model_artifacts"
        saved.mkdir(exist_ok=True)
        for model, booster in models.items():
            booster.save_model(str(saved / f"{model}.txt"))
        joblib.dump(transition, saved / "transition_logistic.joblib", compress=3)
        choice_sets = {name: list(dict.fromkeys(["legacy"] + ([value] if value != "raw" else [])))
                       for name, value in choices.items()}
        testing = []
        predictions = args.output / "student_predictions"
        predictions.mkdir(exist_ok=True)
        for trip in test:
            raw = load(paths[trip])
            serialized = {"reference": raw["reference"], "candidates": raw["candidates"]}
            for model, decoder, ix in all_predictions(raw, models, transition, choice_sets):
                predicted, stats = test_metrics(raw, ix)
                testing.append(dict(trip_id=trip, model=model, decoder=decoder, **stats))
                serialized[f"{model}__{decoder}"] = predicted
            np.savez_compressed(predictions / f"{trip}.npz", **serialized)
            print(f"Student test {trip} complete", flush=True)
    csv_write(args.output / "student_test_per_trip.csv", testing)
    summary = summarize_test(testing)
    csv_write(args.output / "student_test_comparison.csv", summary)
    for row in summary:
        print(row, flush=True)


if __name__ == "__main__":
    main()
