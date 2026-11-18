"""Fit the CPU champion with all 80 classifier training trips."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from pathsnap.io import load_prepared
from pathsnap.pipeline import FILTER_METRES, LGBM_PARAMS, SEED, THREADS, transition_parameters
from pathsnap.training import transition_examples


def fit(trips):
    from lightgbm import LGBMClassifier

    x, y, pair_x, pair_y = [], [], [], []
    excluded = []
    rng = np.random.default_rng(SEED)
    for path in trips:
        data = load_prepared(path)
        labels = data["candidates"] == data["reference"][:, None]
        x.append(data["features"].reshape(-1, 51))
        y.append(labels.ravel())
        # Sample all trips in order to reproduce the verified transition fit's RNG.
        transition, transition_labels = transition_examples(data, labels, rng)
        if data["median_aligned_distance_m"] <= FILTER_METRES:
            pair_x.append(transition)
            pair_y.append(transition_labels)
        else:
            excluded.append(path.stem)
    features, labels = np.concatenate(x), np.concatenate(y)
    del x, y
    with threadpool_limits(limits=THREADS):
        booster = LGBMClassifier(**LGBM_PARAMS).fit(features, labels).booster_
        transition_model = make_pipeline(
            StandardScaler(), LogisticRegression(C=1., max_iter=150, class_weight="balanced"))
        transition_model.fit(np.concatenate(pair_x), np.concatenate(pair_y))
    return booster, transition_parameters(transition_model), excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("prepared"))
    parser.add_argument("--output", type=Path, default=Path("trained-model"))
    args = parser.parse_args()
    trips = sorted((args.prepared / "train").glob("*.npz"))
    if len(trips) != 80:
        parser.error(f"Expected all 80 prepared training trips; found {len(trips)}.")
    booster, transition, excluded = fit(trips)
    args.output.mkdir(exist_ok=True, parents=True)
    booster.save_model(str(args.output / "lightgbm.txt"))
    (args.output / "transition.json").write_text(json.dumps(transition, indent=2)+"\n", encoding="utf-8")
    info = dict(classifier_fit_trips=len(trips), transition_fit_trips=len(trips)-len(excluded),
                transition_excluded_trips=excluded, lightgbm=LGBM_PARAMS,
                dataset_sha256=(args.prepared / "dataset_sha256.txt").read_text().strip())
    (args.output / "training.json").write_text(json.dumps(info, indent=2)+"\n", encoding="utf-8")
    print(f"Saved the retrained champion to {args.output}.")


if __name__ == "__main__":
    main()
