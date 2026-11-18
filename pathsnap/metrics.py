"""Metrics used in the verified integrated experiment."""
import numpy as np
from difflib import SequenceMatcher
from sklearn.metrics import f1_score

from .phase2 import arc_ids


def dedup(road_ids):
    return road_ids[np.r_[True, road_ids[1:] != road_ids[:-1]]] if len(road_ids) else road_ids


def evaluate(selected_idx, candidates, reference, route, arcs, finite, reset):
    prediction = arc_ids(candidates, selected_idx)
    truth = prediction == reference
    assigned = prediction >= 0
    valid = assigned[:-1] & assigned[1:]
    before, after = prediction[:-1][valid], prediction[1:][valid]
    contiguous = (before == after) | (arcs[before, 1] == arcs[after, 0])
    connected_steps = np.flatnonzero(valid & ~reset)
    bounded_connected = finite[connected_steps, selected_idx[connected_steps],
                               selected_idx[connected_steps+1]]
    macro = f1_score(reference, prediction, average="macro", zero_division=0)
    return prediction, dict(
        observations=len(reference), correct=int(truth.sum()), macro_f1=macro,
        top1=float(truth.mean()), direct_continuity=float(contiguous.mean()) if len(contiguous) else 0.,
        bounded_graph_continuity=float(bounded_connected.mean()) if len(bounded_connected) else "",
        reference_route_sequence_ratio=SequenceMatcher(
            None, dedup(route).tolist(), dedup(prediction[assigned]).tolist(), autojunk=False).ratio(),
    )
