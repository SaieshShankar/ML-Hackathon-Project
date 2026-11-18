"""Direction and graph features shared by the PathSnap experiments."""
import numpy as np

from .graph import GAP_RESET_S, BACKTRACK_JITTER_M, search_distances

WINDOWS = (1, 3, 8)
DIRECTION_NAMES = [f"{side}_{quantity}_{window}"
                   for window in WINDOWS for side in ("forward", "backward")
                   for quantity in ("alignment", "weighted_alignment", "confidence")]
TRANSITION_NAMES = (
    "network_displacement_mismatch", "unreachable", "same_directed_arc",
    "exact_reverse", "turn_agreement", "movement_confidence",
    "turn_angle_cosine", "log_elapsed", "reverse_without_turn",
)


def geometry(track, nodes):
    origin = np.median(track[:, :2], axis=0)
    scale = np.pi / 180 * 6_371_008.8 * np.array([np.cos(np.deg2rad(origin[1])), 1.])
    return (track[:, :2] - origin) * scale, (nodes - origin) * scale


def heading(points, times, before, after, breaks):
    movement = points[after] - points[before]
    distance = np.linalg.norm(movement, axis=1)
    elapsed = times[after] - times[before]
    velocity = distance / np.maximum(elapsed, 1.)
    valid = (elapsed > 0) & (breaks[after] == breaks[before])
    confidence = valid * np.clip((velocity - .65) / 3.35, 0., 1.) * np.clip(distance / 8., 0., 1.)
    unit = movement / np.maximum(distance[:, None], 1e-9)
    return unit, confidence, velocity


def point_features(xy, times, directions):
    ix = np.arange(len(xy))
    gaps = np.diff(times)
    breaks = np.r_[0, np.cumsum((gaps <= 0) | (gaps > 30))]
    features = []
    for width in WINDOWS:
        ahead, behind = np.minimum(ix + width, len(ix) - 1), np.maximum(ix - width, 0)
        for left, right in ((ix, ahead), (behind, ix)):
            motion, confidence, _ = heading(xy, times, left, right, breaks)
            similarity = np.sum(directions * motion[:, None], axis=-1)
            features.extend((similarity, similarity * confidence[:, None],
                             np.broadcast_to(confidence[:, None], similarity.shape)))
    ahead, behind = np.minimum(ix + 4, len(ix) - 1), np.maximum(ix - 4, 0)
    center, confidence, velocity = heading(xy, times, behind, ahead, breaks)
    center_match = np.sum(directions * center[:, None], axis=-1)
    emission_prior = (center_match * confidence[:, None]).astype(np.float32)
    return np.stack(features, axis=-1).astype(np.float32), emission_prior, center, confidence, velocity, breaks


def graph_features(xy, nodes, times, arcs, candidates, directions, breaks):
    endpoint_ids = arcs[candidates]
    start, end = np.moveaxis(nodes[endpoint_ids], -2, 0)
    delta = end - start
    length = np.linalg.norm(delta, axis=-1)
    progress = length * np.clip(np.sum((xy[:, None] - start) * delta, axis=-1) /
                                np.maximum(length ** 2, 1e-12), 0., 1.)
    shape = (len(xy) - 1, candidates.shape[1], candidates.shape[1])
    sources = np.broadcast_to(endpoint_ids[:-1, :, 1, None], shape)
    targets = np.broadcast_to(endpoint_ids[1:, None, :, 0], shape)
    network = search_distances(nodes, arcs, sources, targets)
    network += (length - progress)[:-1, :, None] + progress[1:, None, :]
    stay = candidates[:-1, :, None] == candidates[1:, None, :]
    progress_delta = progress[1:, None, :] - progress[:-1, :, None]
    stay_forward = stay & (progress_delta >= -BACKTRACK_JITTER_M)
    network = np.where(stay_forward, np.maximum(progress_delta, 0.), network)
    elapsed = np.diff(times)
    reset = (elapsed <= 0) | (elapsed > GAP_RESET_S)
    finite = np.isfinite(network)
    gps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    mismatch = np.abs(network - gps[:, None, None]) / np.sqrt(np.clip(elapsed, 1., GAP_RESET_S))[:, None, None]

    reversed_arc = np.all(endpoint_ids[:-1, :, None] == endpoint_ids[1:, None, :, ::-1], axis=-1) & ~stay
    incoming = np.arange(len(times) - 1)
    outgoing = incoming + 1
    in_dir, in_conf, _ = heading(xy, times, np.maximum(incoming - 3, 0), incoming, breaks)
    out_dir, out_conf, _ = heading(xy, times, outgoing, np.minimum(outgoing + 3, len(times) - 1), breaks)
    observed_turn = np.sum(in_dir * out_dir, axis=-1)
    turn_confidence = np.minimum(in_conf, out_conf)
    turn_cosine = np.sum(directions[:-1, :, None] * directions[1:, None, :], axis=-1)
    # A turn needs evidence on both sides. At low speed, keep the penalty soft.
    unsupported_reverse = reversed_arc * (1. - turn_confidence[:, None, None] *
                                           np.clip(-observed_turn[:, None, None], 0., 1.))
    transition = np.stack((
        np.log1p(np.minimum(mismatch, 500.)), ~finite, stay, reversed_arc,
        np.abs(turn_cosine - observed_turn[:, None, None]) * turn_confidence[:, None, None],
        np.broadcast_to(turn_confidence[:, None, None], shape),
        turn_cosine, np.broadcast_to(np.log1p(np.clip(elapsed, 0., GAP_RESET_S))[:, None, None], shape),
        unsupported_reverse,
    ), axis=-1).astype(np.float32)
    return mismatch.astype(np.float32), finite, reset, transition
