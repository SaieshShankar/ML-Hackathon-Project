"""Candidate-context features and topology-aware Viterbi for PathSnap."""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import log_expit

from baseline_viterbi import viterbi

CONTEXT_OFFSETS = (-3, -2, -1, 1, 2, 3)
ROUTE_COSTS = np.array([0., 2., 6., 16., 16.])


def shift(candidates, distance, offset, fill):
    ids = np.full_like(candidates, -1)
    distances = np.full(distance.shape, fill, np.float32)
    if offset > 0:
        ids[:-offset], distances[:-offset] = candidates[offset:], distance[offset:]
    else:
        ids[-offset:], distances[-offset:] = candidates[:offset], distance[:offset]
    return ids, distances


def context_features(original, candidates, arcs):
    """The 23 reviewed offline context features, in their original order."""
    distance = original[..., 0]
    fill = float(distance.max()) + 1.
    features, flags, matches = [], [], []
    for offset in CONTEXT_OFFSETS:
        other, distance_other = shift(candidates, distance, offset, fill)
        equal = candidates[..., None] == other[:, None, :]
        seen = equal.any(axis=2)
        close = np.where(seen, (equal * distance_other[:, None, :]).sum(axis=2) /
                         np.maximum(equal.sum(axis=2), 1), fill).astype(np.float32)
        features.extend((seen, close))
        flags.append(seen)
        matches.append(close)
    features.extend((np.mean(flags, axis=0), np.mean(matches, axis=0)))

    arc = candidates[..., None]
    for offset in (-1, 1, -2, 2):
        other, distance_other = shift(candidates, distance, offset, fill)
        valid = other >= 0
        other = np.where(valid, other, 0)[:, None, :]
        joined = arcs[arc, 1] == arcs[other, 0] if offset > 0 else arcs[other, 1] == arcs[arc, 0]
        joined = (joined | (arc == other)) & valid[:, None, :]
        features.extend((np.where(joined, distance_other[:, None, :], fill).min(axis=2),
                         joined.any(axis=2)))
    features.append(distance - distance.mean(axis=1, keepdims=True))
    return np.concatenate((original, np.stack(features, axis=-1).astype(np.float32)), axis=-1)


def topology_classes(candidates, arcs):
    """Vectorised legacy topology edge classes plus a separate exact-reverse mask."""
    current, following = candidates[:-1, :, None], candidates[1:, None, :]
    a, b = arcs[current], arcs[following]
    stay = current == following
    direct = a[..., 1] == b[..., 0]
    reverse = direct & (a[..., 0] == b[..., 1]) & ~stay
    shape = np.broadcast_shapes(a[..., 0].shape, b[..., 0].shape)
    ends = np.broadcast_to(a[..., 1], shape).ravel()
    starts = np.broadcast_to(b[..., 0], shape).ravel()
    graph_size = int(arcs.max()) + 1
    graph = csr_matrix((np.ones(len(arcs), dtype=bool), (arcs[:, 0], arcs[:, 1])),
                       shape=(graph_size, graph_size))
    bridge = np.asarray(graph[ends, starts]).reshape(shape)
    category = np.where(bridge, 2, 3)
    category = np.where(a[..., 0] == b[..., 1], 4, category)
    category = np.where(direct, 1, category)
    category = np.where(stay, 0, category).astype(np.int8)
    return category, reverse


@dataclass(frozen=True)
class Decoder:
    emission: float = .5
    unsupported_reverse: float = 0.
    exact_reverse_cost: float | None = None
    network_weight: float = 0.
    learned_weight: float = 0.
    heading_prior: float = 0.
    reset_gaps: bool = False


DECODERS = {
    "legacy": Decoder(),
    "topology_emission_035": Decoder(emission=.35),
    "topology_emission_075": Decoder(emission=.75),
    "topology_emission_100": Decoder(emission=1.),
    "topology_gap_reset": Decoder(reset_gaps=True),
    "correct_reverse_6": Decoder(exact_reverse_cost=6.),
    "correct_reverse_16": Decoder(exact_reverse_cost=16.),
    "direction_soft": Decoder(unsupported_reverse=1., heading_prior=.25, reset_gaps=True),
    "direction_stronger": Decoder(unsupported_reverse=3., heading_prior=.5, reset_gaps=True),
    "topology_plus_distance": Decoder(network_weight=.65, reset_gaps=True),
    "topology_learned_light": Decoder(learned_weight=.6, heading_prior=.2, reset_gaps=True),
    "topology_learned_strong": Decoder(learned_weight=1.4, heading_prior=.35, reset_gaps=True),
    "topology_learned_no_reset": Decoder(learned_weight=.7),
}


def learned_potentials(model, transition):
    score = model.decision_function(transition.reshape(-1, transition.shape[-1]))
    log = log_expit(score).reshape(transition.shape[:-1]).astype(np.float32)
    return log - log.max(axis=(1, 2), keepdims=True)


def decode(log_prob, category, reverse, data, learned, config):
    costs = ROUTE_COSTS[category]
    if config.exact_reverse_cost is not None:
        costs[reverse] = config.exact_reverse_cost
    if config.unsupported_reverse:
        costs += config.unsupported_reverse * data["transition"][..., 8]
    if config.network_weight:
        costs += config.network_weight * np.minimum(data["mismatch"] / 12., 10.)
    if config.learned_weight:
        costs -= config.learned_weight * learned
    if config.reset_gaps:
        costs[data["reset"]] = 0.
    point = config.emission * log_prob
    if config.heading_prior:
        point += config.heading_prior * data["prior"]
    return viterbi(point, -costs)
