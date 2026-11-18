"""Frozen model inference, display geometry and separate server-side evaluation."""
from collections import Counter
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from sklearn.metrics import f1_score

from .candidates import align_reference
from .direction import geometry
from .graph import SEARCH_LIMIT_M
from .phase2 import arc_ids, path_status, predict
from .pipeline import preprocess


def predict_track(track, nodes, arcs, models, settings):
    valid = (np.isfinite(track).all(axis=1) & (np.abs(track[:, 0]) <= 180)
             & (np.abs(track[:, 1]) <= 90))
    chosen = np.full(len(track), -1, dtype=np.int32)
    status = np.full(len(track), "no_candidate", dtype="<U18")
    if arcs.size:
        boundaries = np.flatnonzero(np.diff(np.r_[False, valid, False]))
        for first, end in boundaries.reshape(-1, 2):
            data = preprocess(track[first:end], nodes, arcs)
            selected = predict(data, *models, settings)
            chosen[first:end] = arc_ids(data["candidates"], selected)
            status[first:end] = path_status(selected, data["finite"], data["elapsed"], settings)
            if first and selected[0] >= 0:
                status[first] = "after_missing"
    return chosen, status, valid


def map_geometry(track, nodes, arcs, chosen, status, valid):
    gps = track[valid][:, [1, 0]].round(7).tolist()
    if not arcs.size or not gps:
        return {"gps": gps, "roads": [], "route": []}

    xy, graph = geometry(track[valid], nodes)
    edges = np.unique(arcs, axis=0)
    length = np.maximum(np.linalg.norm(graph[edges[:, 1]] - graph[edges[:, 0]], axis=1), .01)
    network = csr_matrix((length, (edges[:, 0], edges[:, 1])), shape=(len(nodes), len(nodes)))

    # Find only the intervening connectors for consecutive, matched observations.
    links = {}
    for a, b, state in zip(chosen[:-1], chosen[1:], status[1:]):
        if a >= 0 and b >= 0 and a != b and state == "matched":
            source, target = int(arcs[a, 1]), int(arcs[b, 0])
            if source != target:
                links.setdefault(source, set()).add(target)
    connectors = {}
    for source, targets in links.items():
        distances, previous = dijkstra(network, directed=True, indices=source,
                                       limit=SEARCH_LIMIT_M, return_predecessors=True)
        for target in targets:
            if not np.isfinite(distances[target]):
                continue
            chain = [target]
            while chain[-1] != source:
                chain.append(int(previous[chain[-1]]))
            connectors[source, target] = chain[-2::-1]

    lines, current, prev = [], [], -1
    for arc, state in zip(chosen, status):
        if arc < 0 or state != "matched":
            if current:
                lines.append(current)
            current, prev = [], -1
        if arc < 0 or arc == prev:
            continue
        start, end = map(int, arcs[arc])
        if not current:
            current = [start, end]
        elif current[-1] == start:
            current.append(end)
        elif (current[-1], start) in connectors:
            current.extend(connectors[current[-1], start])
            current.append(end)
        else:
            lines.append(current)
            current = [start, end]
        prev = arc
    if current:
        lines.append(current)

    undirected = np.unique(np.sort(edges, axis=1), axis=0)
    mids = graph[undirected].mean(axis=1)
    distance, _ = cKDTree(xy).query(mids)
    nearby = np.argsort(distance, kind="stable")[:2400]
    roads = nodes[undirected[nearby[distance[nearby] <= 180]]][:, :, ::-1].round(7).tolist()
    route = [nodes[line][:, ::-1].round(7).tolist() for line in lines]
    return {"gps": gps, "roads": roads, "route": route}


def evaluate_prediction(track, nodes, arcs, chosen, valid, route):
    if route is None or not len(route) or not valid.any():
        return None
    if route.ndim != 1 or route.min() < 0 or route.max() >= len(arcs):
        raise ValueError("Reference route file has invalid arc IDs.")
    xy, graph = geometry(track[valid], nodes)
    reference, _ = align_reference(xy, graph[arcs[:, 0]], graph[arcs[:, 1]], route)
    predicted = chosen[valid]
    return {
        "top1": float(np.mean(predicted == reference)),
        "macro_f1": float(f1_score(reference, predicted, average="macro", zero_division=0)),
        "scored_observations": len(reference),
    }


def run_trip(source, trip_id, models, settings):
    started = perf_counter()
    track, nodes, arcs = source.load_inputs(trip_id)
    inference_started = perf_counter()
    chosen, status, valid = predict_track(track, nodes, arcs, models, settings)
    inference_ms = round(1000 * (perf_counter() - inference_started), 1)
    # The inference function has already returned before reference data is opened.
    metrics = evaluate_prediction(track, nodes, arcs, chosen, valid, source.load_reference(trip_id))
    display = map_geometry(track, nodes, arcs, chosen, status, valid)
    return {
        "trip_id": trip_id,
        "mode": source.mode,
        "metrics": metrics,
        "map": display,
        "observations": len(track),
        "matched_observations": int(np.count_nonzero(chosen >= 0)),
        "breaks": {name: int(count) for name, count in Counter(status.tolist()).items()
                   if name not in {"start", "matched"}},
        "inference_ms": inference_ms,
        "elapsed_ms": round(1000 * (perf_counter() - started), 1),
    }
