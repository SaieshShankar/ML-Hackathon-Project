"""Checks for missing states, gap boundaries, disconnected roads and single points."""
import numpy as np

from pathsnap.phase2 import BASE_DECODER, arc_ids, decode, independent, path_status, topology_inputs
from pathsnap.pipeline import preprocess


def check_missing_and_gaps():
    candidates = np.array([[4, 5], [-1, -1], [4, 5], [4, 5], [4, 5]])
    probabilities = np.array([[.9, .1], [np.nan, np.nan], [.1, .9], [.1, .9], [.1, .9]])
    transitions = np.zeros((4, 2, 2))
    finite = np.ones((4, 2, 2), dtype=bool)
    finite[3, 1, 1] = False
    elapsed = np.array([1., 1., 130., 1.])
    path = decode(probabilities, candidates, transitions, transitions, finite, elapsed, BASE_DECODER)
    np.testing.assert_array_equal(arc_ids(candidates, path), [4, -1, 5, 5, 5])
    np.testing.assert_array_equal(path_status(path, finite, elapsed, BASE_DECODER),
                                  ["start", "no_candidate", "after_missing", "gps_gap", "disconnected"])
    np.testing.assert_array_equal(independent(probabilities, candidates), path)
    empty = np.empty((2, 0))
    no_links = np.empty((1, 0, 0))
    np.testing.assert_array_equal(decode(empty, empty, no_links, no_links, no_links.astype(bool),
                                         [1.], BASE_DECODER), [-1, -1])


def check_short_graph():
    track = np.array([[.001, 0., 0.]])
    nodes = np.array([[0., 0.], [.01, 0.]])
    arcs = np.array([[0, 1]], dtype=int)
    data = preprocess(track, nodes, arcs)
    assert data["features"].shape == (1, 1, 51)
    np.testing.assert_array_equal(data["candidates"], [[0]])
    topology = topology_inputs(data)
    assert topology.shape == (0, 1, 1)
    chosen = decode(np.array([[.9]]), data["candidates"], topology, topology,
                    data["finite"], data["elapsed"], BASE_DECODER)
    np.testing.assert_array_equal(chosen, [0])


def check_gap_cost():
    ids = np.array([[7, 8], [7, 8]])
    scores = np.array([[.99, .01], [.1, .9]])
    topology = np.array([[[0., 4.], [4., 0.]]])
    learned = np.zeros_like(topology)
    finite = np.ones_like(topology, dtype=bool)
    short = decode(scores, ids, topology, learned, finite, np.array([1.]), BASE_DECODER)
    long = decode(scores, ids, topology, learned, finite, np.array([10_000.]), BASE_DECODER)
    np.testing.assert_array_equal(short, [0, 0])
    np.testing.assert_array_equal(long, [0, 1])


if __name__ == "__main__":
    check_missing_and_gaps()
    check_short_graph()
    check_gap_cost()
    print("Passed decoder checks for missing states, gap boundaries, disconnected roads and single-point input.")
