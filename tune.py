"""Run CPU Bayesian tuning and paired whole-trip evaluation for PathSnap."""
import argparse
from dataclasses import asdict
from difflib import SequenceMatcher
from hashlib import file_digest
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from pathsnap.bayes import choose_next, designs
from pathsnap.io import archive_trips, load_prepared, write_rows, zip_trip
from pathsnap.metrics import dedup
from pathsnap.phase2 import BASE_DECODER, DecoderSettings, arc_ids, decode, independent, path_status, topology_inputs
from pathsnap.pipeline import FILTER_METRES, LGBM_PARAMS, SEED, THREADS, preprocess, transition_log, transition_parameters
from pathsnap.training import transition_examples

MODEL_SPACE = {
    "n_estimators": (160, 250, 340),
    "learning_rate": (.045, .06, .085),
    "num_leaves": (31, 63, 95),
    "min_child_samples": (25, 40, 70),
    "reg_lambda": (1., 3.),
}
UNTUNED = {k: LGBM_PARAMS[k] for k in MODEL_SPACE}
MODEL_WARM = [
    UNTUNED,
    dict(n_estimators=250, learning_rate=.085, num_leaves=95, min_child_samples=25, reg_lambda=1.),
    dict(n_estimators=340, learning_rate=.045, num_leaves=31, min_child_samples=70, reg_lambda=3.),
]
DECODER_SPACE = {
    "emission": (.35, .5, .7),
    "learned": (.45, .7, 1.0),
    "gap_seconds": (30., 60., 120.),
    "gap_power": (.25, .5, 1.),
    "unreachable_penalty": (0., 1.5, 3.),
}
DECODER_WARM = [
    asdict(BASE_DECODER),
    asdict(DecoderSettings(.35, 1., 120., .25, 0.)),
    asdict(DecoderSettings(.7, .45, 30., 1., 3.)),
]


def load_inputs(prepared, archive, prefixes):
    blocks = {}
    rng = np.random.default_rng(SEED)
    for trip in sorted(prefixes):
        data = load_prepared(prepared / "train" / f"{trip}.npz")
        if "elapsed" not in data:
            with archive.open(prefixes[trip]+".track") as stream:
                times = np.loadtxt(stream, usecols=2, ndmin=1)
            data["elapsed"] = np.diff(times)
        labels = data["candidates"] == data["reference"][:, None]
        # Draw in the original order even for trips excluded from transition fitting.
        pairs = transition_examples(data, labels, rng)
        retained = data["median_aligned_distance_m"] <= FILTER_METRES
        blocks[trip] = dict(
            data={k: data[k] for k in ("candidates", "reference", "route", "arcs", "finite",
                                       "reset", "elapsed", "transition")},
            x=data["features"].reshape(-1, 51), y=labels.ravel(), pairs=pairs if retained else None,
        )
    return blocks


def train_transition(blocks, fit):
    pairs = [blocks[trip]["pairs"] for trip in fit if blocks[trip]["pairs"] is not None]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=150, class_weight="balanced"))
    model.fit(np.concatenate([x for x, _ in pairs]), np.concatenate([y for _, y in pairs]))
    return transition_parameters(model)


def prime(data, params):
    setup_start = perf_counter()
    transition = transition_log(data["transition"], params)
    topology = topology_inputs(data)
    setup_ms = (perf_counter()-setup_start)*1000
    return {"data": {k: v for k, v in data.items() if k != "transition"},
            "transition": transition, "topology": topology, "setup_ms": setup_ms}


def construct_folds(blocks, lgb):
    ids = np.asarray(sorted(blocks))
    assignments, folds, validation = [], [], {}
    for number, (fit_ix, val_ix) in enumerate(GroupKFold(n_splits=5).split(ids, groups=ids), 1):
        fit, val = ids[fit_ix].tolist(), ids[val_ix].tolist()
        assert len(fit) == 64 and len(val) == 16 and not set(fit) & set(val)
        features = np.concatenate([blocks[trip]["x"] for trip in fit])
        labels = np.concatenate([blocks[trip]["y"] for trip in fit])
        dataset = lgb.Dataset(features, label=labels, params={"seed": SEED, "feature_pre_filter": False,
                                                              "num_threads": THREADS})
        dataset.construct()
        params = train_transition(blocks, fit)
        for trip in val:
            validation[trip] = prime(blocks[trip]["data"], params)
            validation[trip]["fold"] = number
        folds.append((dataset, val))
        assignments += [dict(fold=number, trip_id=trip, role=role)
                        for role, trips in (("fit", fit), ("validation", val)) for trip in trips]
        print(f"Prepared fold {number}/5: fit={len(fit)}, validation={len(val)} trips", flush=True)
    # The all-training fit uses retained features and transition examples after the trials.
    for block in blocks.values():
        del block["data"]
    return folds, validation, assignments


def native_params(params):
    return dict(objective="binary", metric="None", verbosity=-1, seed=SEED, device_type="cpu",
                num_threads=THREADS, force_col_wise=True, deterministic=True,
                bagging_fraction=LGBM_PARAMS["subsample"], bagging_freq=LGBM_PARAMS["subsample_freq"],
                feature_fraction=LGBM_PARAMS["colsample_bytree"], feature_pre_filter=False,
                learning_rate=params["learning_rate"], num_leaves=params["num_leaves"],
                min_data_in_leaf=params["min_child_samples"], lambda_l2=params["reg_lambda"])


def predict_prob(model, features, candidate_shape):
    start = perf_counter()
    probabilities = model.predict(features, num_threads=THREADS).reshape(candidate_shape)
    return probabilities, (perf_counter()-start)*1000


def decode_trip(trip, probabilities, settings):
    data = trip["data"]
    return decode(probabilities, data["candidates"], trip["topology"], trip["transition"],
                  data["finite"], data["elapsed"], settings)


def objective(validation, scores, settings):
    by_fold = [[] for _ in range(5)]
    for trip, data in validation.items():
        candidates = data["data"]["candidates"]
        chosen = arc_ids(candidates, decode_trip(data, scores[trip][0], settings))
        by_fold[data["fold"]-1].append(f1_score(data["data"]["reference"], chosen,
                                                average="macro", zero_division=0))
    averages = [float(np.mean(values)) for values in by_fold]
    return float(np.mean(averages)), averages


def train_trial(blocks, folds, params, lgb):
    scores = {}
    times = []
    for fold, (dataset, validation) in enumerate(folds, 1):
        start = perf_counter()
        model = lgb.train(native_params(params), dataset, num_boost_round=params["n_estimators"])
        times.append(perf_counter()-start)
        for trip in validation:
            block = blocks[trip]
            shape = (len(block["y"])//8, 8)
            scores[trip] = predict_prob(model, block["x"], shape)
        print(f"  fit {fold}/5 {times[-1]:.1f}s", flush=True)
    return scores, times


def search_models(blocks, folds, validation, count, output, lgb):
    options, matrix = designs(MODEL_SPACE)
    visited, history, timing = [], [], []
    baseline_scores, best = None, None
    for index in range(count):
        ei = ""
        if index < len(MODEL_WARM):
            params, mode = MODEL_WARM[index], "baseline" if index == 0 else "initial_design"
        else:
            params, ei = choose_next(options, matrix, visited, SEED)
            mode = "bayesian_expected_improvement"
        print(f"Classifier trial {index+1}/{count} {mode} {params}", flush=True)
        scores, times = train_trial(blocks, folds, params, lgb)
        f1, fold_values = objective(validation, scores, BASE_DECODER)
        visited.append((params, f1))
        history.append(dict(trial=index+1, selection=mode, expected_improvement=ei, **params,
                            cv_macro_f1=f1, fit_seconds=sum(times),
                            **{f"fold_{i}": val for i, val in enumerate(fold_values, 1)}))
        timing += [dict(trial=index+1, fold=i, seconds=value) for i, value in enumerate(times, 1)]
        if index == 0:
            baseline_scores = scores
        if best is None or f1 > best[0]:
            best = f1, params, scores, index+1
        write_rows(output / "classifier_bayes_trials.csv", history)
        write_rows(output / "cv_fit_timings.csv", timing)
        print(f"  decoded CV Macro F1 {f1:.6f}", flush=True)
    return baseline_scores, best


def search_decoder(validation, scores, count, output):
    options, matrix = designs(DECODER_SPACE)
    visited, rows = [], []
    for index in range(count):
        ei = ""
        if index < len(DECODER_WARM):
            params = DECODER_WARM[index]
            mode = "baseline" if index == 0 else "initial_design"
        else:
            params, ei = choose_next(options, matrix, visited, SEED)
            mode = "bayesian_expected_improvement"
        f1, folds = objective(validation, scores, DecoderSettings(**params))
        visited.append((params, f1))
        rows.append(dict(trial=index+1, selection=mode, expected_improvement=ei, **params, cv_macro_f1=f1,
                         **{f"fold_{i}": value for i, value in enumerate(folds, 1)}))
        write_rows(output / "decoder_bayes_trials.csv", rows)
        print(f"Decoder trial {index+1}/{count}: CV F1={f1:.6f} {params}", flush=True)
    return max(rows, key=lambda row: row["cv_macro_f1"])


def metric_row(trip, probabilities, indices, settings):
    data = trip["data"]
    candidates, reference, arcs = data["candidates"], data["reference"], data["arcs"]
    chosen = arc_ids(candidates, indices)
    assigned = chosen >= 0
    valid_pair = assigned[:-1] & assigned[1:]
    before, after = chosen[:-1][valid_pair], chosen[1:][valid_pair]
    direct = (before == after) | (arcs[before, 1] == arcs[after, 0])
    status = path_status(indices, data["finite"], data["elapsed"], settings)
    ground = candidates == reference[:, None]
    p, r, f1, _ = precision_recall_fscore_support(reference, chosen, average="macro", zero_division=0)
    wrong = assigned & (chosen != reference)
    return chosen, status, dict(
        observations=len(chosen), correct=int((chosen == reference).sum()),
        macro_f1=f1, macro_precision=p, macro_recall=r,
        candidate_log_loss=log_loss(ground.ravel(), probabilities.ravel(), labels=[False, True]),
        miss_candidates=int((~ground.any(axis=1)).sum()), no_candidate=int((~assigned).sum()),
        direct_continuity=float(direct.mean()) if len(direct) else 0.,
        route_sequence_similarity=SequenceMatcher(
            None, dedup(data["route"]).tolist(), dedup(chosen[assigned]).tolist(),
            autojunk=False).ratio() if assigned.any() else 0.,
        reverse_errors=int(np.all(arcs[chosen[wrong]] == arcs[reference[wrong]][:, ::-1], axis=1).sum()),
        gap_breaks=int((status == "gps_gap").sum()), disconnected_breaks=int((status == "disconnected").sum()),
    )


def score_split(split, trips, model_scores, settings, output):
    outcomes = []
    prediction_dir = output / "predictions" / split
    prediction_dir.mkdir(exist_ok=True, parents=True)
    for trip, entry in sorted(trips.items()):
        candidates = entry["data"]["candidates"]
        preds = {"reference": entry["data"]["reference"], "candidates": candidates}
        for variant, scores in model_scores.items():
            probabilities, classifier_ms = scores[trip]
            preds[variant+"_probabilities"] = probabilities
            config = settings[variant]
            for mode in ("classifier_only", "viterbi"):
                start = perf_counter()
                indices = (decode_trip(entry, probabilities, config) if mode == "viterbi"
                           else independent(probabilities, candidates))
                decode_ms = (perf_counter()-start)*1000 if mode == "viterbi" else 0.
                chosen, status, row = metric_row(entry, probabilities, indices, config)
                outcomes.append(dict(split=split, trip_id=trip, fold=entry.get("fold", 0), model=variant,
                                     mode=mode, **row, classifier_ms=classifier_ms, decode_ms=decode_ms,
                                     transition_setup_ms=entry["setup_ms"] if mode == "viterbi" else 0.))
                preds[f"{variant}_{mode}"] = chosen
                if mode == "viterbi":
                    preds[f"{variant}_path_status"] = status
        np.savez_compressed(prediction_dir / f"{trip}.npz", **preds)
    return outcomes


def aggregate(rows):
    output = []
    for split in ("cv", "student_test"):
        for variant in ("untuned", "tuned"):
            for mode in ("classifier_only", "viterbi"):
                group = [r for r in rows if (r["split"], r["model"], r["mode"]) == (split, variant, mode)]
                weights = np.array([r["observations"] for r in group])
                f1 = [np.mean([r["macro_f1"] for r in group if r["fold"] == fold])
                      for fold in sorted({r["fold"] for r in group})]
                output.append(dict(
                    split=split, model=variant, mode=mode, trips=len(group), observations=int(weights.sum()),
                    macro_f1=float(np.mean(f1)), macro_f1_fold_sd=np.std(f1, ddof=1) if len(f1) > 1 else "",
                    top1=sum(r["correct"] for r in group)/weights.sum(),
                    macro_precision=np.mean([r["macro_precision"] for r in group]),
                    macro_recall=np.mean([r["macro_recall"] for r in group]),
                    candidate_log_loss=np.average([r["candidate_log_loss"] for r in group], weights=weights),
                    candidate_miss_rate=sum(r["miss_candidates"] for r in group)/weights.sum(),
                    direct_continuity=np.mean([r["direct_continuity"] for r in group]),
                    route_sequence_similarity=np.mean([r["route_sequence_similarity"] for r in group]),
                    reverse_errors=sum(r["reverse_errors"] for r in group),
                    gap_breaks=sum(r["gap_breaks"] for r in group),
                    disconnected_breaks=sum(r["disconnected_breaks"] for r in group),
                    no_candidate=sum(r["no_candidate"] for r in group),
                    median_classifier_ms=np.median([r["classifier_ms"] for r in group]),
                    median_viterbi_ms=np.median([r["decode_ms"] for r in group]),
                    median_transition_setup_ms=np.median([r["transition_setup_ms"] for r in group]),
                    median_scoring_and_decoder_ms=np.median(
                        [r["classifier_ms"]+r["decode_ms"]+r["transition_setup_ms"] for r in group]),
                ))
    return output


def final_evaluation(archive, prefixes, blocks, tuning, decoders, output, lgb):
    features = np.concatenate([b["x"] for b in blocks.values()])
    labels = np.concatenate([b["y"] for b in blocks.values()])
    dataset = lgb.Dataset(features, label=labels, params={"seed": SEED, "feature_pre_filter": False,
                                                          "num_threads": THREADS})
    dataset.construct()
    del features, labels
    transition = train_transition(blocks, list(blocks))
    models, timings = {}, []
    for variant, params in tuning.items():
        start = perf_counter()
        models[variant] = lgb.train(native_params(params), dataset, num_boost_round=params["n_estimators"])
        timings.append(dict(model=variant, training_trips=len(blocks), fit_seconds=perf_counter()-start))
        print(f"Refit {variant} in {timings[-1]['fit_seconds']:.1f}s on {len(blocks)} trips", flush=True)
        model_dir = output / "models" / variant
        model_dir.mkdir(exist_ok=True, parents=True)
        models[variant].save_model(str(model_dir / "lightgbm.txt"))
        (model_dir / "transition.json").write_text(json.dumps(transition, indent=2)+"\n", encoding="utf-8")
        (model_dir / "decoder.json").write_text(json.dumps(asdict(decoders[variant]), indent=2)+"\n", encoding="utf-8")
    write_rows(output / "refit_timings.csv", timings)
    trips, predictions = {}, {variant: {} for variant in tuning}
    for trip, prefix in sorted(prefixes.items()):
        raw = preprocess(*zip_trip(archive, prefix))
        for variant, model in models.items():
            predictions[variant][trip] = predict_prob(
                model, raw["features"].reshape(-1, 51), raw["candidates"].shape)
        del raw["features"]
        trips[trip] = prime(raw, transition)
    return score_split("student_test", trips, predictions, decoders, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "prepared", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--classifier-trials", type=int, default=8)
    parser.add_argument("--decoder-trials", type=int, default=12)
    args = parser.parse_args()
    if args.classifier_trials <= len(MODEL_WARM) or args.decoder_trials <= len(DECODER_WARM):
        parser.error("Each search requires at least one adaptive Bayesian trial.")
    import lightgbm as lgb

    with args.dataset.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    assert digest == (args.prepared / "dataset_sha256.txt").read_text().strip()
    args.output.mkdir(exist_ok=True, parents=True)
    with ZipFile(args.dataset) as archive, threadpool_limits(limits=THREADS):
        prefixes = archive_trips(archive)
        blocks = load_inputs(args.prepared, archive, prefixes["train"])
        excluded = [trip for trip, block in blocks.items() if block["pairs"] is None]
        folds, validation, assignments = construct_folds(blocks, lgb)
        write_rows(args.output / "fold_assignments.csv", assignments)
        baseline_scores, (_, trial_params, tuned_scores, model_trial) = search_models(
            blocks, folds, validation, args.classifier_trials, args.output, lgb)
        del folds
        decoder_trial = search_decoder(validation, tuned_scores, args.decoder_trials, args.output)
        chosen = DecoderSettings(**{k: decoder_trial[k] for k in DECODER_SPACE})
        decoders = {"untuned": BASE_DECODER, "tuned": chosen}
        classifiers = {"untuned": UNTUNED, "tuned": trial_params}
        manifest = dict(dataset_sha256=digest, classifiers=classifiers,
                        decoders={k: asdict(v) for k, v in decoders.items()},
                        selected_classifier_trial=model_trial, selected_decoder_trial=decoder_trial["trial"],
                        transition_excluded_trips=excluded, baseline_classifier_trials=args.classifier_trials,
                        decoder_trials=args.decoder_trials, threads=THREADS, device="cpu",
                        lightgbm=lgb.__version__, selection_objective="mean decoded observation Macro F1 across 5 whole-trip folds")
        (args.output / "frozen_selection.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        print("Frozen selection before opening the ten student-test trips:", json.dumps(manifest), flush=True)
        outcomes = score_split("cv", validation, {"untuned": baseline_scores, "tuned": tuned_scores},
                               decoders, args.output)
        del validation, baseline_scores, tuned_scores
        outcomes += final_evaluation(archive, prefixes["student_test"], blocks, classifiers,
                                     decoders, args.output, lgb)
    write_rows(args.output / "per_trip_metrics.csv", outcomes)
    summary = aggregate(outcomes)
    write_rows(args.output / "comparison.csv", summary)
    for row in summary:
        print(row, flush=True)


if __name__ == "__main__":
    main()
