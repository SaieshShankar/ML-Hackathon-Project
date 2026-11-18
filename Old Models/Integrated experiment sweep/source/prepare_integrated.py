"""Rebuild integrated features from the raw ZIP and original five-model cache."""
import argparse
from hashlib import file_digest
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np

from baseline_viterbi import load_raw
from direction_features import geometry, graph_features, point_features


def build_trip(archive, split, trip_id, baseline):
    track, nodes, arcs, route = load_raw(archive, split, trip_id)
    assert np.array_equal(arcs, baseline["arcs"])
    candidates = baseline["candidates"]
    xy, graph = geometry(track, nodes)
    delta = graph[arcs[candidates, 1]] - graph[arcs[candidates, 0]]
    directions = delta / np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1e-9)
    more, prior, _, _, _, breaks = point_features(xy, track[:, 2], directions)
    mismatch, finite, reset, transition = graph_features(
        xy, graph, track[:, 2], arcs, candidates, directions, breaks)
    return dict(original=baseline["x"], more=more, prior=prior, candidates=candidates,
                reference=baseline["reference"], route=route, arcs=arcs, finite=finite,
                mismatch=mismatch, reset=reset, transition=transition)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("dataset", "cache", "set1", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--only-trip", help="Build a single trip for a smoke check.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.dataset.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != (args.set1 / "dataset_sha256.txt").read_text().strip():
            raise ValueError("Raw archive does not match the baseline dataset digest")
    cache = joblib.load(args.cache)
    with ZipFile(args.dataset) as archive:
        for split, subset in (("cv", "train"), ("student_test", "student_test")):
            for trip in sorted(cache[subset]):
                if args.only_trip and trip != args.only_trip:
                    continue
                data = build_trip(archive, split, trip, cache[subset][trip])
                np.savez(args.output / f"{trip}.npz", **data)
                print(f"Prepared {split} trip {trip}", flush=True)


if __name__ == "__main__":
    main()
