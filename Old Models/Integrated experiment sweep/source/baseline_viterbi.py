"""Compare saved Histogram Gradient Boosting scores with graph-aware Viterbi."""
import argparse
import csv
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from hashlib import file_digest
from itertools import product
import json
from pathlib import Path
import platform
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import sklearn
from sklearn.metrics import f1_score


SEARCH_LIMIT_M = 2000
GAP_RESET_S = 60
BACKTRACK_JITTER_M = 10
BASELINE = "hist_gb"
MODEL = "Histogram Gradient Boosting"


@dataclass(frozen=True)
class Settings:
    scale_m: int
    transition_weight: float
    unreachable_cost: int
    emission: str

    @property
    def name(self):
        return f"{self.emission}_s{self.scale_m}_w{self.transition_weight:g}_u{self.unreachable_cost}"


# Fixed before reading student-test outputs.
GRID = [Settings(*params) for params in product((12, 30), (.5, 1., 2.), (5, 10), ("logp", "logodds"))]


def csv_read(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_raw(archive, split, trip):
    directory = "train" if split == "cv" else "student_test"
    prefix = f"Student folder/{directory}/{trip}/{trip}"
    def load(extension, dtype, ndmin=2):
        with archive.open(f"{prefix}.{extension}") as handle:
            return np.loadtxt(handle, dtype=dtype, ndmin=ndmin)
    return load("track", float), load("nodes", float), load("arcs", np.int32), load("route", np.int32, 1)


def search_distances(nodes, arcs, sources, targets):
    """Bounded directed shortest paths from each previous arc end to next arc start."""
    count = len(nodes)
    endpoints = np.unique(arcs, axis=0)
    weights = np.maximum(np.linalg.norm(nodes[endpoints[:, 1]] - nodes[endpoints[:, 0]], axis=1), .01)
    graph = coo_matrix((weights, (endpoints[:, 0], endpoints[:, 1])), shape=(count, count)).tocsr()

    # Run once per distinct source per trip and retain only requested pairs.
    keys = sources.astype(np.int64) * count + targets
    distinct, inverse = np.unique(keys.ravel(), return_inverse=True)
    end, start = np.divmod(distinct, count)
    source_ids, group_starts = np.unique(end, return_index=True)
    group_ends = np.r_[group_starts[1:], len(distinct)]
    found = np.empty(len(distinct))
    batch_size = min(32, max(1, 2_000_000 // count))
    for i in range(0, len(source_ids), batch_size):
        last = min(i + batch_size, len(source_ids))
        distances = dijkstra(graph, directed=True, indices=source_ids[i:last],
                             limit=SEARCH_LIMIT_M, return_predecessors=False)
        begin, finish = group_starts[i], group_ends[last - 1]
        local_rows = np.repeat(np.arange(last - i), group_ends[i:last] - group_starts[i:last])
        found[begin:finish] = distances[local_rows, start[begin:finish]]
    return found[inverse].reshape(keys.shape)


def transition_inputs(track, nodes, arcs, candidates):
    origin = np.median(track[:, :2], axis=0)
    metres = np.pi / 180 * 6_371_008.8 * np.array([np.cos(np.deg2rad(origin[1])), 1.])
    xy = (track[:, :2] - origin) * metres
    nodes = (nodes - origin) * metres
    endpoint_ids = arcs[candidates]
    start, end = np.moveaxis(nodes[endpoint_ids], -2, 0)
    delta = end - start
    length = np.linalg.norm(delta, axis=-1)
    fraction = np.clip(np.sum((xy[:, None] - start) * delta, axis=-1) /
                       np.maximum(length ** 2, 1e-12), 0., 1.)
    progress = length * fraction
    shape = (len(track) - 1, candidates.shape[1], candidates.shape[1])
    sources = np.broadcast_to(endpoint_ids[:-1, :, 1, None], shape)
    targets = np.broadcast_to(endpoint_ids[1:, None, :, 0], shape)
    network = search_distances(nodes, arcs, sources, targets)
    network += (length - progress)[:-1, :, None] + progress[1:, None, :]
    same = candidates[:-1, :, None] == candidates[1:, None, :]
    delta_progress = progress[1:, None, :] - progress[:-1, :, None]
    can_stay = same & (delta_progress >= -BACKTRACK_JITTER_M)
    network = np.where(can_stay, np.maximum(delta_progress, 0), network)

    displacement = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    elapsed = np.diff(track[:, 2])
    reset = (elapsed <= 0) | (elapsed > GAP_RESET_S)
    finite = np.isfinite(network)
    # A bounded mismatch metric is converted to a log transition potential.
    mismatch = np.abs(network - displacement[:, None, None]) / np.sqrt(np.clip(elapsed, 1., GAP_RESET_S))[:, None, None]
    return mismatch, finite, reset


def viterbi(emissions, transitions):
    """Decode a candidate sequence with log-space dynamic programming."""
    n, k = emissions.shape
    parents = np.empty((n, k), dtype=np.int16)
    columns = np.arange(k)
    score = emissions[0].copy()
    for t in range(1, n):
        joint = score[:, None] + transitions[t - 1]
        previous = joint.argmax(axis=0)
        parents[t] = previous
        score = joint[previous, columns] + emissions[t]
        score -= score.max()
    selected = np.empty(n, dtype=np.int16)
    selected[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        selected[t - 1] = parents[t, selected[t]]
    return selected


def decode(scores, mismatch, finite, reset, setting):
    q = np.clip(scores, 1e-7, 1. - 1e-7)
    emissions = np.log(q)
    if setting.emission == "logodds":
        emissions -= np.log1p(-q)
    costs = np.where(finite, np.minimum(mismatch / setting.scale_m, setting.unreachable_cost),
                     float(setting.unreachable_cost))
    costs[reset] = 0.
    return viterbi(emissions, -setting.transition_weight * costs)


def dedup(road_ids):
    return road_ids[np.r_[True, road_ids[1:] != road_ids[:-1]]]


def evaluate(selected_idx, candidates, reference, route, arcs, finite, reset):
    prediction = candidates[np.arange(len(reference)), selected_idx]
    truth = prediction == reference
    contiguous = ((prediction[:-1] == prediction[1:]) |
                  (arcs[prediction[:-1], 1] == arcs[prediction[1:], 0]))
    bounded_connected = finite[np.arange(len(reference) - 1), selected_idx[:-1], selected_idx[1:]]
    macro = f1_score(reference, prediction, average="macro", zero_division=0)
    return prediction, dict(
        observations=len(reference), correct=int(truth.sum()), macro_f1=macro,
        top1=float(truth.mean()), direct_continuity=float(contiguous.mean()),
        bounded_graph_continuity=float(bounded_connected[~reset].mean()) if np.any(~reset) else "",
        reference_route_sequence_ratio=SequenceMatcher(
            None, dedup(route).tolist(), dedup(prediction).tolist(), autojunk=False).ratio(),
    )


def aggregate(rows):
    fold_groups = {}
    for row in rows:
        fold_groups.setdefault(row["fold"], []).append(row["macro_f1"])
    fold_means = np.array([np.mean(group) for group in fold_groups.values()])
    return dict(
        trips=len(rows), observations=sum(r["observations"] for r in rows),
        correct=sum(r["correct"] for r in rows),
        macro_f1=float(fold_means.mean()),
        macro_f1_fold_sd=float(fold_means.std(ddof=1)) if len(fold_means) > 1 else "",
        top1=sum(r["correct"] for r in rows) / sum(r["observations"] for r in rows),
        direct_continuity=float(np.mean([r["direct_continuity"] for r in rows])),
        bounded_graph_continuity=float(np.mean([r["bounded_graph_continuity"] for r in rows if r["bounded_graph_continuity"] != ""])),
        reference_route_sequence_ratio=float(np.mean([r["reference_route_sequence_ratio"] for r in rows])),
        median_transition_setup_ms=float(np.median([r["transition_setup_ms"] for r in rows])),
        median_decode_ms=float(np.median([r["decode_ms"] for r in rows])),
        median_decoder_total_ms=float(np.median([r["transition_setup_ms"] + r["decode_ms"] for r in rows])),
    )


def run_split(archive, evidence, output, split, baselines, configurations):
    results = {cfg.name: [] for cfg in configurations}
    results[BASELINE] = []
    paths = sorted(evidence.glob(f"predictions/histogram_gradient_boosting_{split}_*.npz"))
    expected = 80 if split == "cv" else 10
    if len(paths) != expected:
        raise ValueError(f"Expected {expected} prediction archives for {split}, found {len(paths)}")
    digest_rows = []
    cases = output / "student_predictions"
    if split == "student_test":
        cases.mkdir(exist_ok=True)
    cv_cache = output / "work_cv_predictions"
    if split == "cv":
        cv_cache.mkdir(exist_ok=True)
    for index, path in enumerate(paths, 1):
        trip_id = path.stem.rsplit("_", 1)[-1]
        original = baselines[split, trip_id]
        with np.load(path) as saved:
            reference, candidates, scores, saved_prediction = (
                saved[name] for name in ("reference", "candidate_arcs", "scores", "selected"))
        with path.open("rb") as handle:
            digest_rows.append(dict(file=path.name, sha256=file_digest(handle, "sha256").hexdigest()))
        base_idx = scores.argmax(axis=1)
        track, nodes, arcs, route = load_raw(archive, split, trip_id)
        assert len(track) == len(reference)
        start = perf_counter()
        mismatch, finite, reset = transition_inputs(track, nodes, arcs, candidates)
        setup_ms = 1000 * (perf_counter() - start)
        start = perf_counter()
        baseline_pred, baseline_stats = evaluate(base_idx, candidates, reference, route, arcs, finite, reset)
        assert np.array_equal(saved_prediction, baseline_pred)
        assert np.isclose(baseline_stats["macro_f1"], float(original["observation_macro_f1"]))
        assert np.isclose(baseline_stats["top1"], float(original["top1_accuracy"]))
        assert np.isclose(baseline_stats["direct_continuity"], float(original["direct_continuity"]))
        info = dict(split=split, fold=int(original["fold"]), trip_id=trip_id,
                    gap_resets=int(reset.sum()), candidate_miss_rate=float(original["candidate_miss_rate"]))
        results[BASELINE].append(dict(setting=BASELINE, **info, **baseline_stats,
                                      transition_setup_ms=0., decode_ms=0., recovered=0, regressed=0))
        cv_paths = {}
        for setting in configurations:
            start = perf_counter()
            selected_idx = decode(scores, mismatch, finite, reset, setting)
            decode_ms = 1000 * (perf_counter() - start)
            if split == "cv":
                # Only calculate sequence metrics for the prespecified baseline and
                # the selected setting, after the parameter search is complete.
                prediction = candidates[np.arange(len(reference)), selected_idx]
                stats = dict(macro_f1=f1_score(reference, prediction, average="macro", zero_division=0))
            else:
                prediction, stats = evaluate(selected_idx, candidates, reference, route, arcs, finite, reset)
            recovered = int(np.sum((prediction == reference) & (baseline_pred != reference)))
            regressed = int(np.sum((prediction != reference) & (baseline_pred == reference)))
            results[setting.name].append(dict(setting=setting.name, **info, **stats,
                                              transition_setup_ms=setup_ms, decode_ms=decode_ms,
                                              recovered=recovered, regressed=regressed))
            if split == "student_test":
                np.savez_compressed(cases / f"{trip_id}.npz", reference=reference,
                                    decoded=prediction, classifier=baseline_pred,
                                    candidate_arcs=candidates)
            else:
                cv_paths[setting.name] = selected_idx
        if split == "cv":
            np.savez_compressed(cv_cache / f"{trip_id}.npz", finite=finite, reset=reset, **cv_paths)
        print(f"{split} {index}/{expected} trip={trip_id} observations={len(track)} graph={setup_ms:.0f} ms", flush=True)
    csv_write(output / f"{split}_prediction_digests.csv", digest_rows)
    return results


def choose_and_score_cv(archive, evidence, output, results):
    search = []
    for setting in GRID:
        rows = results[setting.name]
        folds = [np.mean([row["macro_f1"] for row in rows if row["fold"] == i]) for i in range(1, 6)]
        search.append(dict(setting=setting.name, **asdict(setting), cv_macro_f1=np.mean(folds),
                           cv_fold_sd=np.std(folds, ddof=1), **{f"fold_{i}": f for i, f in enumerate(folds, 1)}))
    search.sort(key=lambda row: (-row["cv_macro_f1"], row["setting"]))
    csv_write(output / "decoder_search.csv", search)
    winner = next(setting for setting in GRID if setting.name == search[0]["setting"])
    print(f"Selected {winner.name} on CV Macro F1 {search[0]['cv_macro_f1']:.6f}", flush=True)
    rows = results[winner.name]
    for row in rows:
        trip_id = row["trip_id"]
        path = evidence / "predictions" / f"histogram_gradient_boosting_cv_{trip_id}.npz"
        with np.load(path) as saved:
            reference, candidates = saved["reference"], saved["candidate_arcs"]
        path = output / "work_cv_predictions" / f"{trip_id}.npz"
        with np.load(path) as saved:
            selected_idx, finite, reset = saved[winner.name], saved["finite"], saved["reset"]
        _, _, arcs, route = load_raw(archive, "cv", trip_id)
        _, stats = evaluate(selected_idx, candidates, reference, route, arcs, finite, reset)
        assert np.isclose(stats["macro_f1"], row["macro_f1"])
        row.update(stats)
    print("Verified all 80 out-of-fold winner prediction metrics.", flush=True)
    return winner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.dataset.open("rb") as handle:
        dataset_hash = file_digest(handle, "sha256").hexdigest()
    assert dataset_hash == (args.evidence / "dataset_sha256.txt").read_text().strip()
    baselines = {(row["split"], row["trip_id"]): row
                 for row in csv_read(args.evidence / "per_trip_metrics.csv") if row["model"] == MODEL}
    with ZipFile(args.dataset) as archive:
        cv = run_split(archive, args.evidence, args.output, "cv", baselines, GRID)
        winner = choose_and_score_cv(archive, args.evidence, args.output, cv)
        (args.output / "experiment.json").write_text(json.dumps(dict(
            selected_setting=asdict(winner), selected_name=winner.name,
            dataset_sha256=dataset_hash, python=platform.python_version(),
            numpy=np.__version__, scipy=scipy.__version__, scikit_learn=sklearn.__version__,
            gap_reset_s=GAP_RESET_S, graph_search_limit_m=SEARCH_LIMIT_M,
            backward_jitter_m=BACKTRACK_JITTER_M,
        ), indent=2), encoding="utf-8")
        test = run_split(archive, args.evidence, args.output, "student_test", baselines, [winner])
    summary = []
    all_rows = []
    for split, runs in (("cv", cv), ("student_test", test)):
        for setting in (BASELINE, winner.name):
            rows = runs[setting]
            all_rows.extend(rows)
            summary.append(dict(split=split, setting=setting, **aggregate(rows)))
    csv_write(args.output / "per_trip_metrics.csv", all_rows)
    csv_write(args.output / "comparison.csv", summary)
    print("Final comparison:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
