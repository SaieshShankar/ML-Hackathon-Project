"""Reproducible PathSnap candidate-classification benchmark."""
import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import joblib
import numpy as np
import sklearn
import scipy
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 352
FOLDS = 5
K = 8
SHORTLIST = 64
FIT_STRIDE = 2
THREADS = 2
SKIP_COST = 2.0
FEATURES = [
    "log_point_distance", "projection_fraction", "forward_heading_alignment",
    "backward_heading_alignment", "forward_motion_m", "backward_motion_m",
    "log_segment_length", "log_midpoint_distance", "log_nearest_distance_gap",
    "candidate_rank",
]


def projection(points, starts, ends):
    delta = ends - starts
    offset = points - starts
    denominator = np.maximum(np.sum(delta * delta, axis=-1), 1e-12)
    fraction = np.clip(np.sum(offset * delta, axis=-1) / denominator, 0, 1)
    residual = offset - fraction[..., None] * delta
    return np.linalg.norm(residual, axis=-1), fraction


def motion(points):
    delta = np.diff(points, axis=0)
    length = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(length[:, None], 1e-8)
    return (
        np.vstack((direction, np.zeros(2))), np.vstack((np.zeros(2), direction)),
        np.r_[length, 0], np.r_[0, length],
    )


def candidates(points, starts, ends):
    midpoints = (starts + ends) / 2
    _, shortlist = cKDTree(midpoints).query(points, k=SHORTLIST, workers=THREADS)
    distance, fraction = projection(points[:, None], starts[shortlist], ends[shortlist])
    order = np.argsort(distance, axis=1, kind="stable")[:, :K]
    ids = np.take_along_axis(shortlist, order, axis=1).astype(np.int32)
    distance = np.take_along_axis(distance, order, axis=1)
    fraction = np.take_along_axis(fraction, order, axis=1)
    segments = ends[ids] - starts[ids]
    length = np.linalg.norm(segments, axis=-1)
    direction = segments / np.maximum(length[..., None], 1e-8)
    forward, backward, forward_m, backward_m = motion(points)
    features = np.stack(
        [
            np.log1p(distance), fraction,
            np.sum(direction * forward[:, None], axis=-1),
            np.sum(direction * backward[:, None], axis=-1),
            np.broadcast_to(np.log1p(forward_m[:, None]), ids.shape),
            np.broadcast_to(np.log1p(backward_m[:, None]), ids.shape),
            np.log1p(length),
            np.log1p(np.linalg.norm(points[:, None] - midpoints[ids], axis=-1)),
            np.log1p(np.maximum(distance - distance[:, :1], 0)),
            np.broadcast_to(np.arange(K), ids.shape),
        ],
        axis=-1,
    ).astype(np.float32)
    return ids, features, distance


def align_reference(points, starts, ends, route):
    """Minimum-cost monotone alignment to the ordered reference arc positions.

    At each observation pay point-to-segment distance. Staying and advancing one
    position are free; each skipped intermediate position adds SKIP_COST metres.
    The first and last positions may be anywhere in the reference route.
    """
    n, m = len(points), len(route)
    positions = np.arange(m, dtype=np.int32)
    route_starts, route_ends = starts[route], ends[route]
    parents = np.empty((n, m), dtype=np.int32)
    previous = None
    for start in range(0, n, 128):
        costs, _ = projection(points[start:start + 128, None], route_starts, route_ends)
        for offset, cost in enumerate(costs):
            index = start + offset
            if index == 0:
                previous = cost
                continue
            prefix = previous - SKIP_COST * positions
            best_prefix = np.minimum.accumulate(prefix)
            new_minimum = np.r_[True, prefix[1:] < best_prefix[:-1]]
            prefix_parent = np.maximum.accumulate(np.where(new_minimum, positions, 0))
            advance = best_prefix[:-1] + SKIP_COST * positions[:-1]
            move = advance < previous[1:]
            parent = positions.copy()
            parent[1:] = np.where(move, prefix_parent[:-1], positions[1:])
            previous = cost + np.r_[previous[0], np.where(move, advance, previous[1:])]
            parents[index] = parent

    assigned_positions = np.empty(n, dtype=np.int32)
    assigned_positions[-1] = previous.argmin()
    for index in range(n - 1, 0, -1):
        assigned_positions[index - 1] = parents[index, assigned_positions[index]]
    return route[assigned_positions], assigned_positions


def read_trip(archive, folder, trip_id, split):
    def read(extension, dtype, dimensions=2):
        with archive.open(f"{folder}/{trip_id}.{extension}") as stream:
            return np.loadtxt(stream, dtype=dtype, ndmin=dimensions)

    track = read("track", np.float64)
    nodes = read("nodes", np.float64)
    arcs = read("arcs", np.int32)
    route = read("route", np.int32, 1)
    # Local equirectangular approximation with trip-centred longitude scaling.
    origin = np.median(track[:, :2], axis=0)
    scale = (np.pi / 180) * 6_371_008.8 * np.array([np.cos(np.deg2rad(origin[1])), 1])
    points = (track[:, :2] - origin) * scale
    graph = (nodes - origin) * scale
    starts, ends = graph[arcs[:, 0]], graph[arcs[:, 1]]
    ids, features, distances = candidates(points, starts, ends)
    reference, aligned_positions = align_reference(points, starts, ends, route)
    reference_distance, _ = projection(points, starts[reference], ends[reference])
    missed = ~np.any(ids == reference[:, None], axis=1)
    sample = np.unique(np.linspace(0, len(track) - 1, 12, dtype=int))
    audited = 0
    for index in sample:
        exact, _ = projection(points[index], starts, ends)
        audited += int(distances[index, -1] <= np.partition(exact, K - 1)[K - 1] + 1e-7)
    audit = dict(
        split=split, trip_id=trip_id, observations=len(track), nodes=len(nodes),
        road_arcs=len(arcs), reference_arcs=len(route), candidate_misses=int(missed.sum()),
        exact_top8_distance_audits_passed=audited, exact_top8_distance_audits=len(sample),
        median_aligned_distance_m=float(np.median(reference_distance)),
        p95_aligned_distance_m=float(np.quantile(reference_distance, .95)),
        aligned_start_route_index=int(aligned_positions[0]),
        aligned_end_route_index=int(aligned_positions[-1]),
        time_gaps_over_5_s=int(np.sum(np.diff(track[:, 2]) > 5)),
        nonincreasing_timestamps=int(np.sum(np.diff(track[:, 2]) <= 0)),
    )
    return dict(x=features, candidates=ids, reference=reference, arcs=arcs, audit=audit)


def prepare(dataset, cache, output):
    started = perf_counter()
    data = {"train": {}, "student_test": {}}
    with ZipFile(dataset) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith(".track"):
                root, split, trip_id, _ = name.split("/")
                if split in data:
                    data[split][trip_id] = read_trip(archive, f"{root}/{split}/{trip_id}", trip_id, split)
                    print(f"Prepared {split} {trip_id}", flush=True)
    if [len(split) for split in data.values()] != [80, 10]:
        raise ValueError("Expected 80 training and 10 student-test trips")
    joblib.dump(data, cache, compress=1)
    write_rows(output / "dataset_audit.csv", [t["audit"] for s in data.values() for t in s.values()])
    print(f"Preparation and cache completed in {perf_counter()-started:.1f} s", flush=True)


def model_specs():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(
            C=1, max_iter=250, random_state=SEED)),
        "Shallow Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=30, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=32, max_depth=12, min_samples_leaf=12, n_jobs=THREADS, random_state=SEED),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=75, max_leaf_nodes=31, learning_rate=.1, early_stopping=False, random_state=SEED),
        "Shallow MLP": make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(32,), batch_size=2048, learning_rate_init=.002, max_iter=12,
            random_state=SEED)),
    }


def training_xy(trips, trip_ids):
    feature_blocks, label_blocks = [], []
    for index, trip_id in enumerate(trip_ids):
        trip = trips[trip_id]
        # Keep one in every two observations per trip, alternating start parity.
        rows = slice(index % FIT_STRIDE, None, FIT_STRIDE)
        feature_blocks.append(trip["x"][rows].reshape(-1, len(FEATURES)))
        label_blocks.append((trip["candidates"][rows] == trip["reference"][rows, None]).ravel())
    return np.concatenate(feature_blocks), np.concatenate(label_blocks)


def evaluate(model, trip, model_name, split, fold, trip_id, prediction_dir):
    candidates = trip["candidates"]
    reference = trip["reference"]
    y = candidates == reference[:, None]
    started = perf_counter()
    if model is None:
        selected = candidates[:, 0]
        scores = None
    else:
        scores = model.predict_proba(trip["x"].reshape(-1, len(FEATURES)))[:, 1].reshape(-1, K)
        selected = candidates[np.arange(len(reference)), scores.argmax(axis=1)]
    latency = (perf_counter() - started) * 1000

    top_pair = candidates == selected[:, None]
    precision, recall, pair_f1, _ = precision_recall_fscore_support(
        y.ravel(), top_pair.ravel(), average="binary", zero_division=0)
    # Direct forward adjacency. Remaining on the same arc is continuous.
    arcs = trip["arcs"]
    contiguous = (selected[:-1] == selected[1:]) | (arcs[selected[:-1], 1] == arcs[selected[1:], 0])
    result = dict(
        model=model_name, split=split, fold=fold, trip_id=trip_id, observations=len(reference),
        observation_macro_f1=f1_score(reference, selected, average="macro", zero_division=0),
        top1_accuracy=float(np.mean(reference == selected)),
        candidate_pair_f1=pair_f1, candidate_pair_precision=precision, candidate_pair_recall=recall,
        candidate_pair_log_loss="" if scores is None else log_loss(y.ravel(), scores.ravel(), labels=[False, True]),
        candidate_miss_rate=float(np.mean(~y.any(axis=1))),
        direct_continuity=float(np.mean(contiguous)),
        scoring_ms=latency,
    )
    if prediction_dir:
        key = model_name.lower().replace(" ", "_")
        np.savez_compressed(prediction_dir / f"{key}_{split}_{trip_id}.npz",
                            selected=selected, reference=reference, candidate_arcs=candidates,
                            scores=np.empty(0) if scores is None else scores)
    return result


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(cache, output, selected):
    data = joblib.load(cache)
    train = data["train"]
    test = data["student_test"]
    training_ids = np.asarray(sorted(train))
    all_models = model_specs()
    models = all_models if selected is None else {name: all_models[name] for name in selected}
    models = {"Nearest segment (heuristic)": None, **models}
    folds = list(GroupKFold(FOLDS).split(training_ids, groups=training_ids))
    write_rows(output / "fold_assignments.csv", [
        {"fold": i, "trip_id": trip, "role": role}
        for i, (fit, validation) in enumerate(folds, 1)
        for role, ix in (("fit", fit), ("validation", validation))
        for trip in training_ids[ix]
    ])
    model_params = {
        name: ({key: value for key, value in model.get_params().items()
               if isinstance(value, (bool, int, float, str, type(None)))}
               if model is not None else {})
        for name, model in models.items()
    }
    (output / "run_config.json").write_text(json.dumps(dict(
        argv=sys.argv, python=platform.python_version(), platform=platform.platform(),
        numpy=np.__version__, scipy=scipy.__version__, scikit_learn=sklearn.__version__,
        seed=SEED, fold_count=FOLDS, shortlist=SHORTLIST, candidates_per_observation=K,
        fit_stride=FIT_STRIDE, threads=THREADS, skip_cost_m=SKIP_COST,
        features=FEATURES, model_parameters=model_params,
    ), indent=2), encoding="utf-8")

    # Keep predictions for all held-out and student-test observations.
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    artifact_dir = output / "models"
    artifact_dir.mkdir(exist_ok=True)
    metrics, timings, loss_rows = [], [], []
    stages = [
        ("cv", fold, training_ids[fit], train, training_ids[validation])
        for fold, (fit, validation) in enumerate(folds, 1)
    ] + [("student_test", 0, training_ids, test, sorted(test))]
    with threadpool_limits(limits=THREADS):
        for split, fold, fit_ids, evaluation_trips, eval_ids in stages:
            x, y = training_xy(train, fit_ids)
            print(f"{split} fold {fold}: {len(fit_ids)} fit trips, {len(x)} candidate rows", flush=True)
            for name, model in models.items():
                started = perf_counter()
                if model is not None:
                    model.fit(x, y)
                fit_seconds = perf_counter() - started if model is not None else 0
                if name == "Shallow MLP":
                    loss_rows.extend(
                        dict(split=split, fold=fold, epoch=i, training_loss=loss)
                        for i, loss in enumerate(model[-1].loss_curve_, 1)
                    )
                start_count = len(metrics)
                for trip_id in eval_ids:
                    metrics.append(evaluate(
                        model, evaluation_trips[trip_id], name, split, fold, trip_id, prediction_dir))
                timings.append(dict(model=name, split=split, fold=fold, fit_seconds=fit_seconds,
                                    fit_trips=len(fit_ids), fit_candidate_rows=len(x)))
                if split == "student_test" and model is not None:
                    joblib.dump(model, artifact_dir / (name.lower().replace(" ", "_") + ".joblib"), compress=3)
                batch_f1 = np.mean([row["observation_macro_f1"] for row in metrics[start_count:]])
                print(f"  {name}: fit={fit_seconds:.2f}s, Macro F1={batch_f1:.5f}", flush=True)
                write_rows(output / "per_trip_metrics.csv", metrics)
                write_rows(output / "fit_timings.csv", timings)
                if loss_rows:
                    write_rows(output / "mlp_training_loss.csv", loss_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run"])
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", choices=list(model_specs()))
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    if args.stage == "prepare":
        digest = hashlib.file_digest(args.dataset.open("rb"), "sha256").hexdigest()
        (args.output / "dataset_sha256.txt").write_text(digest + "\n", encoding="ascii")
        prepare(args.dataset, args.cache, args.output)
    else:
        run(args.cache, args.output, args.model)


if __name__ == "__main__":
    main()
