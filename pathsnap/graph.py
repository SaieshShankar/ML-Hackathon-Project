"""Shortest-path utilities and Viterbi decoding used by the champion."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

SEARCH_LIMIT_M = 2000
GAP_RESET_S = 60
BACKTRACK_JITTER_M = 10


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
