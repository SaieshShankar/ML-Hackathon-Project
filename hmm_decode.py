"""PathSnap HMM / Viterbi route reconstruction on top of benchmark.py outputs.

Hidden state   : the K=8 candidate arcs at each GPS observation (arc ids kept).
Emission       : alpha * log(classifier score of the candidate).
Transition     : cost from the road graph. Moving from arc a to arc b costs
                 hop_cost * (number of arc hops from a to b), 0 if the same arc.
                 Not reachable within MAX_HOPS -> unreachable_cost (finite, so the
                 decoder never gets stuck on GPS gaps or missing candidates).
                 Reversing onto the opposite-direction arc adds uturn_cost.
Parameters are tuned on the 80 training trips (held-out CV predictions) only;
student-test trips are reported, not used for tuning. Evaluator trips are never touched.

Usage (after benchmark.py prepare + run):
    python hmm_decode.py --cache data.joblib --run run --model histogram_gradient_boosting
"""
import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

MAX_HOPS = 4


def make_reach(arcs, depth=MAX_HOPS):
    """reach(a) -> {arc: hops} for arcs reachable from arc a by following end->start nodes."""
    outgoing = defaultdict(list)
    for index, (start, _) in enumerate(arcs):
        outgoing[int(start)].append(index)
    memo = {}

    def reach(a):
        if a in memo:
            return memo[a]
        seen, frontier = {a: 0}, [a]
        for hop in range(1, depth + 1):
            following = []
            for x in frontier:
                for y in outgoing.get(int(arcs[x, 1]), []):
                    if y not in seen:
                        seen[y] = hop
                        following.append(y)
            frontier = following
        memo[a] = seen
        return seen

    return reach


def transition_tables(trip):
    """Parameter-independent hop counts H[t,i,j] and u-turn flags U[t,i,j] (t -> t+1)."""
    cand, arcs = trip["candidates"], trip["arcs"]
    reach = make_reach(arcs)
    n, k = cand.shape
    hops = np.full((n - 1, k, k), MAX_HOPS + 1, dtype=np.int8)
    uturn = np.zeros((n - 1, k, k), dtype=bool)
    for t in range(n - 1):
        for i in range(k):
            a = int(cand[t, i])
            table = reach(a)
            for j in range(k):
                b = int(cand[t + 1, j])
                if b in table:
                    hops[t, i, j] = table[b]
                if b != a and arcs[b, 0] == arcs[a, 1] and arcs[b, 1] == arcs[a, 0]:
                    uturn[t, i, j] = True
    return hops, uturn


def viterbi(scores, hops, uturn, alpha, hop_cost, unreachable_cost, uturn_cost):
    n, k = scores.shape
    emission = alpha * np.log(np.clip(scores, 1e-6, 1.0))
    reachable = hops <= MAX_HOPS
    cost = np.where(reachable, hop_cost * hops, unreachable_cost) + uturn_cost * uturn
    delta = emission[0].copy()
    back = np.zeros((n, k), dtype=np.int8)
    for t in range(1, n):
        total = delta[:, None] - cost[t - 1]          # (from, to)
        back[t] = total.argmax(axis=0)
        delta = total.max(axis=0) + emission[t]
    path = np.empty(n, dtype=np.int64)
    path[-1] = delta.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def load_trips(cache, run, model, split):
    data = joblib.load(cache)
    trips = data["train"] if split == "cv" else data["student_test"]
    out = {}
    for trip_id, trip in trips.items():
        file = run / "predictions" / f"{model}_{split}_{trip_id}.npz"
        if not file.exists():
            continue
        saved = np.load(file)
        assert np.array_equal(saved["candidate_arcs"], trip["candidates"]), "candidate mismatch"
        hops, uturn = transition_tables(trip)
        out[trip_id] = dict(scores=saved["scores"], cand=trip["candidates"], ref=trip["reference"],
                            arcs=trip["arcs"], hops=hops, uturn=uturn)
    return out


def evaluate(trips, params):
    rows = []
    for trip_id, t in trips.items():
        base = t["cand"][np.arange(len(t["ref"])), t["scores"].argmax(axis=1)]
        path = viterbi(t["scores"], t["hops"], t["uturn"], **params)
        dec = t["cand"][np.arange(len(t["ref"])), path]
        def cont(sel):
            return float(np.mean((sel[:-1] == sel[1:]) | (t["arcs"][sel[:-1], 1] == t["arcs"][sel[1:], 0])))
        rows.append(dict(trip_id=trip_id, n=len(dec),
                         clf_top1=float(np.mean(base == t["ref"])), hmm_top1=float(np.mean(dec == t["ref"])),
                         clf_cont=cont(base), hmm_cont=cont(dec)))
    return rows


def mean_by_obs(rows, key):
    w = np.array([r["n"] for r in rows])
    return float(np.average([r[key] for r in rows], weights=w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--model", default="histogram_gradient_boosting")
    args = ap.parse_args()

    print("Loading CV (training-trip) predictions...")
    cv = load_trips(args.cache, args.run, args.model, "cv")
    print(f"  {len(cv)} trips. Precomputing graph transitions done.")

    grid = dict(alpha=[0.5, 1.0, 2.0], hop_cost=[0.5, 1.0, 2.0, 3.0],
                unreachable_cost=[4.0, 8.0, 12.0, 20.0], uturn_cost=[0.0, 3.0, 8.0])
    best, trials = None, []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        rows = evaluate(cv, params)
        score = mean_by_obs(rows, "hmm_top1")
        trials.append({**params, "cv_hmm_top1": score})
        if best is None or score > best[0]:
            best = (score, params)
    cv_rows = evaluate(cv, best[1])
    print(f"\nBest decoder params (chosen on training trips only): {best[1]}")
    print(f"CV top-1  classifier-only: {mean_by_obs(cv_rows, 'clf_top1'):.4f}   HMM: {mean_by_obs(cv_rows, 'hmm_top1'):.4f}")

    st = load_trips(args.cache, args.run, args.model, "student_test")
    st_rows = evaluate(st, best[1])
    print(f"Student top-1  classifier-only: {mean_by_obs(st_rows, 'clf_top1'):.4f}   HMM: {mean_by_obs(st_rows, 'hmm_top1'):.4f}")
    print(f"Student continuity  classifier-only: {mean_by_obs(st_rows, 'clf_cont'):.4f}   HMM: {mean_by_obs(st_rows, 'hmm_cont'):.4f}")

    out = args.run
    (out / "hmm_config.json").write_text(json.dumps(dict(model=args.model, max_hops=MAX_HOPS, params=best[1]), indent=2))
    for name, rows in (("hmm_cv_per_trip.csv", cv_rows), ("hmm_student_per_trip.csv", st_rows), ("hmm_trials.csv", trials)):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"Saved hmm_config.json and CSVs in {out}")


if __name__ == "__main__":
    main()
