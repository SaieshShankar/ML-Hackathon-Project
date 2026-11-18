"""Shared preprocessing and decoding for the selected PathSnap model."""
import json
from pathlib import Path

import numpy as np
from scipy.special import log_expit

from .candidates import align_reference, candidates, projection
from .context import context_features, topology_classes
from .direction import geometry, graph_features, point_features
from .graph import viterbi

THREADS = 4
SEED = 352
FILTER_METRES = 50.
LGBM_PARAMS = dict(n_estimators=250, learning_rate=.06, num_leaves=63,
                   min_child_samples=40, subsample=.8, subsample_freq=1,
                   colsample_bytree=.8, reg_lambda=1., device_type="cpu",
                   n_jobs=THREADS, verbose=-1, random_state=SEED,
                   deterministic=True, force_col_wise=True)
ROUTE_COSTS = np.array([0., 2., 6., 16., 16.])
EMISSION_WEIGHT = .5
LEARNED_WEIGHT = .7
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def default_model_dir():
    tuned = MODEL_DIR / "phase2-tuned"
    if all((tuned / name).is_file() for name in ("lightgbm.txt", "transition.json", "decoder.json")):
        return tuned
    return MODEL_DIR


def preprocess(track, nodes, arcs, route=None):
    xy, graph = geometry(track, nodes)
    start, end = graph[arcs[:, 0]], graph[arcs[:, 1]]
    ids, original, _ = candidates(xy, start, end)
    delta = end[ids] - start[ids]
    directions = delta / np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1e-9)
    more, _, _, _, _, breaks = point_features(xy, track[:, 2], directions)
    _, finite, reset, transition = graph_features(xy, graph, track[:, 2], arcs, ids,
                                                  directions, breaks)
    features = np.concatenate((context_features(original, ids, arcs), more), axis=-1)
    data = dict(candidates=ids, features=features, arcs=arcs, finite=finite, reset=reset,
                elapsed=np.diff(track[:, 2]),
                transition=transition)
    if route is not None:
        reference, _ = align_reference(xy, start, end, route)
        distance, _ = projection(xy, start[reference], end[reference])
        data.update(route=route, reference=reference,
                    median_aligned_distance_m=float(np.median(distance)))
    return data


def load_models(directory):
    from lightgbm import Booster

    booster = Booster(model_file=str(directory / "lightgbm.txt"))
    transition = json.loads((directory / "transition.json").read_text(encoding="utf-8"))
    return booster, transition


def transition_parameters(model):
    scale, logistic = model.steps[0][1], model.steps[1][1]
    return dict(mean=scale.mean_.tolist(), scale=scale.scale_.tolist(),
                coef=logistic.coef_[0].tolist(), intercept=float(logistic.intercept_[0]))


def predict_indices(data, booster, parameters):
    ids = data["candidates"]
    probability = booster.predict(data["features"].reshape(-1, 51),
                                  num_threads=THREADS).reshape(ids.shape)
    emissions = EMISSION_WEIGHT * np.log(np.clip(probability, 1e-6, 1.))
    learned = transition_log(data["transition"], parameters)
    classes, _ = topology_classes(ids, data["arcs"])
    costs = ROUTE_COSTS[classes] - LEARNED_WEIGHT * learned
    return viterbi(emissions, -costs)


def transition_log(transition, parameters):
    normalized = transition.reshape(-1, transition.shape[-1]).copy()
    normalized -= np.asarray(parameters["mean"])
    normalized /= np.asarray(parameters["scale"])
    decision = normalized @ np.asarray(parameters["coef"]) + parameters["intercept"]
    learned = log_expit(decision).reshape(transition.shape[:-1]).astype(np.float32)
    learned -= learned.max(axis=(1, 2), keepdims=True)
    return learned
