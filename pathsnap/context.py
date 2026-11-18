"""Adjacent GPS candidate features and directed topology classes."""
import numpy as np
from scipy.sparse import csr_matrix

CONTEXT_OFFSETS = (-3, -2, -1, 1, 2, 3)


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
    if len(candidates) < 2 or candidates.shape[1] == 0:
        shape = (max(0, len(candidates)-1), candidates.shape[1], candidates.shape[1])
        return np.empty(shape, dtype=np.int8), np.empty(shape, dtype=bool)
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
