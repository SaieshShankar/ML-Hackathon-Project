"""Robust graph-constrained HMM-style decoding for PathSnap Phase 2."""
from dataclasses import dataclass
import json

import numpy as np

from .context import topology_classes
from .graph import viterbi
from .pipeline import ROUTE_COSTS, THREADS, transition_log


@dataclass(frozen=True)
class DecoderSettings:
    emission: float = .5
    learned: float = .7
    gap_seconds: float = 60.
    gap_power: float = .5
    unreachable_penalty: float = 0.


BASE_DECODER = DecoderSettings()


def load_decoder(directory):
    path = directory / "decoder.json"
    return DecoderSettings(**json.loads(path.read_text(encoding="utf-8"))) if path.exists() else BASE_DECODER


def predict(data, booster, parameters, settings):
    candidates = data["candidates"]
    probabilities = booster.predict(data["features"].reshape(-1, 51),
                                  num_threads=THREADS).reshape(candidates.shape)
    return decode(probabilities, candidates, topology_inputs(data),
                  transition_log(data["transition"], parameters),
                  data["finite"], data["elapsed"], settings)


def topology_inputs(data):
    classes, _ = topology_classes(data["candidates"], data["arcs"])
    return ROUTE_COSTS[classes].astype(np.float32)


def decode(probabilities, candidates, topology, log_learned, finite, elapsed, settings):
    """Return candidate indices, with -1 where an observation has no usable state."""
    valid = (candidates >= 0) & np.isfinite(probabilities)
    usable = valid.any(axis=1)
    selected = np.full(len(candidates), -1, dtype=np.int16)
    if not np.any(usable):
        return selected
    emission = np.where(valid, settings.emission * np.log(np.clip(probabilities, 1e-6, 1.)), -np.inf)
    costs = topology.astype(np.float64) - settings.learned * log_learned
    if settings.unreachable_penalty:
        costs += settings.unreachable_penalty * ~finite
    break_time = ~np.isfinite(elapsed) | (elapsed <= 0)
    costs[break_time] = 0.
    large = elapsed > settings.gap_seconds
    if settings.gap_power and np.any(large):
        weight = (settings.gap_seconds / elapsed[large]) ** settings.gap_power
        costs[large] *= weight[:, None, None]
    boundaries = np.flatnonzero(np.diff(np.r_[False, usable, False]))
    for start, end in boundaries.reshape(-1, 2):
        selected[start:end] = viterbi(emission[start:end], -costs[start:end - 1])
    return selected


def independent(probabilities, candidates):
    if candidates.shape[1] == 0:
        return np.full(len(candidates), -1, dtype=np.int16)
    valid = (candidates >= 0) & np.isfinite(probabilities)
    order = np.argmax(np.where(valid, probabilities, -np.inf), axis=1)
    return np.where(valid.any(axis=1), order, -1)


def arc_ids(candidates, indices):
    if candidates.shape[1] == 0:
        return np.full(len(indices), -1, dtype=np.int32)
    return np.where(indices >= 0, candidates[np.arange(len(indices)), np.maximum(indices, 0)], -1)


def path_status(indices, finite, elapsed, settings):
    """Flag gaps and unreachable choices so downstream maps can break route lines."""
    status = np.full(len(indices), "matched", dtype="<U18")
    if len(indices) == 0:
        return status
    status[0] = "start"
    status[indices < 0] = "no_candidate"
    valid = (indices[:-1] >= 0) & (indices[1:] >= 0)
    t = np.arange(len(indices)-1)
    if finite.shape[1]:
        reachable = finite[t, np.maximum(indices[:-1], 0), np.maximum(indices[1:], 0)]
        status[1:][valid & ~reachable] = "disconnected"
    gap = (~np.isfinite(elapsed)) | (elapsed <= 0) | (elapsed > settings.gap_seconds)
    status[1:][valid & gap] = "gps_gap"
    status[1:][(indices[:-1] < 0) & (indices[1:] >= 0)] = "after_missing"
    return status
