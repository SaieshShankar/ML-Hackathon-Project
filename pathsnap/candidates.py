"""Generate the original eight candidates and training reference labels."""
import numpy as np
from scipy.spatial import cKDTree

K = 8
SHORTLIST = 64
THREADS = 2
SKIP_COST = 2.0


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
    _, shortlist = cKDTree(midpoints).query(
        points, k=min(SHORTLIST, len(starts)), workers=THREADS)
    shortlist = shortlist.reshape(len(points), -1)
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
            np.broadcast_to(np.arange(ids.shape[1]), ids.shape),
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
