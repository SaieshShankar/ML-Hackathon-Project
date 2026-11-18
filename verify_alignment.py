import numpy as np
from benchmark import align_reference, projection, SKIP_COST

rng = np.random.default_rng(352)
for n, m in [(1, 4), (4, 1), (9, 13)]:
    for _ in range(20):
        starts = rng.uniform(0, 40, (m, 2))
        ends = starts + rng.uniform(-10, 10, (m, 2))
        points = rng.uniform(0, 40, (n, 2))
        route = rng.permutation(m)
        _, assigned = align_reference(points, starts, ends, route)
        cost, _ = projection(points[:, None], starts[route], ends[route])
        chosen = cost[np.arange(n), assigned].sum()
        chosen += SKIP_COST * np.maximum(np.diff(assigned) - 1, 0).sum()
        oracle = cost[0].copy()
        for scores in cost[1:]:
            oracle = scores + np.array([
                min(oracle[k] + SKIP_COST * max(j - k - 1, 0) for k in range(j + 1))
                for j in range(m)
            ])
        assert np.all(np.diff(assigned) >= 0)
        assert np.isclose(chosen, oracle.min())
print("Verified all 60 deterministic synthetic alignments against exhaustive transitions.")
