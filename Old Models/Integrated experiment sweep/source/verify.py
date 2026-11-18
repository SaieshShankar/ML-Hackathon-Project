"""Verify recorded metrics and produce the integrated LightGBM result summary."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from baseline_viterbi import csv_read, csv_write
from methods import DECODERS

LABELS = {
    "context_clean": "LightGBM + context, filtered",
    "context_heading_clean": "LightGBM + context + heading, filtered",
    "context_heading_all": "LightGBM + context + heading, all trips",
}


def verify(run, set1, prepared):
    config = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    assert config["device"] == "cpu" and config["lightgbm_params"]["device_type"] == "cpu"
    with (set1 / "fold_assignments.csv").open("rb") as stream:
        assert config["fold_assignments_sha256"] == file_digest(stream, "sha256").hexdigest()
    fold_assignments = csv_read(set1 / "fold_assignments.csv")
    validation = {r["trip_id"]: int(r["fold"]) for r in fold_assignments if r["role"] == "validation"}
    fit = {i: {r["trip_id"] for r in fold_assignments if r["role"] == "fit" and int(r["fold"]) == i}
           for i in range(1, 6)}
    sweep = csv_read(run / "cv_search.csv")
    cv = csv_read(run / "cv_per_trip.csv")
    assert len(sweep) == len(LABELS) * (len(DECODERS)+1)
    assert len(cv) == 80 * len(sweep)
    assert all(validation[r["trip_id"]] == int(r["fold"]) for r in cv)
    assert len({(r["trip_id"], r["model"], r["decoder"]) for r in cv}) == len(cv)
    winner = max(sweep, key=lambda r: float(r["macro_f1_all"]))
    assert winner["model"] == config["selected_by_cv"]["model"]
    assert winner["decoder"] == config["selected_by_cv"]["decoder"]
    for row in sweep:
        scores = [r for r in cv if r["model"] == row["model"] and r["decoder"] == row["decoder"]]
        f1 = float(np.mean([float(r["macro_f1"]) for r in scores]))
        assert np.isclose(f1, float(row["macro_f1_all"]))
    for row in csv_read(run / "fit_timings.csv"):
        fold = int(row["fold"])
        trips = fit[fold] if fold else set(validation)
        count = len(trips) if row["model"] == "context_heading_all" else len(trips-set(config["excluded"]))
        assert count == int(row["trips"])
    per_trip = csv_read(run / "student_test_per_trip.csv")
    results_by_trip = {}
    for row in per_trip:
        results_by_trip.setdefault(row["trip_id"], {})[(row["model"], row["decoder"])] = row
    assert len(results_by_trip) == 10 and set(results_by_trip).isdisjoint(validation)
    for trip, expected in sorted(results_by_trip.items()):
        with np.load(run / "student_predictions" / f"{trip}.npz") as archive, np.load(prepared / f"{trip}.npz") as cache:
            candidates, reference, arcs = archive["candidates"], archive["reference"], cache["arcs"]
            assert np.array_equal(candidates, cache["candidates"])
            assert np.array_equal(reference, cache["reference"])
            for (model, decoder), row in expected.items():
                path = archive[f"{model}__{decoder}"]
                assert np.all((candidates == path[:, None]).any(axis=1))
                assert np.isclose(f1_score(reference, path, average="macro"), float(row["macro_f1"]))
                assert int((path == reference).sum()) == int(row["correct"])
                wrong = path != reference
                reversed_arc = wrong & np.all(arcs[path] == arcs[reference][:, ::-1], axis=1)
                assert int(reversed_arc.sum()) == int(row["reverse_errors"])
        print(f"Verified {len(expected)} predictions for {trip}", flush=True)
    print("Verified fold membership, fit-time filtering, CV selection, and all student-test scores.")
    return config, sweep, per_trip


def write_report(run, comparisons, config, sweep, per_trip):
    results = csv_read(run / "student_test_comparison.csv")
    choices = config["decoder_choice_per_model"]
    winning = config["selected_by_cv"]
    bench = csv_read(comparisons / "pathsnap-better-review" / "decoder_crossover.csv")
    better = next(r for r in bench if r["pipeline"] == "lgbm_topology")
    earlier = csv_read(comparisons / "pathsnap-direction" / "comparison.csv")
    prior = max((r for r in earlier if r["split"] == "student_test"), key=lambda r: float(r["macro_f1"]))
    prior_cv = next(r for r in earlier if r["split"] == "cv" and r["pipeline"] == prior["pipeline"])
    key = lambda row: (row["model"], row["decoder"])
    cv = {key(r): r for r in sweep}
    board = [
        dict(pipeline="Previous in-house best (XGBoost Ranker + direction rules)",
             decoder="direction rules", cv_macro_f1=float(prior_cv["macro_f1"]),
             student_macro_f1=float(prior["macro_f1"]), student_top1=float(prior["top1"]),
             reverse_errors=prior["reverse_errors"], trips_at_090="", selected_by_cv=False),
        dict(pipeline="Attached LightGBM (replayed)", decoder="legacy", cv_macro_f1=.8664127659856045,
             student_macro_f1=float(better["macro_f1"]), student_top1=float(better["top1_micro"]),
             reverse_errors="", trips_at_090=3, selected_by_cv=False),
    ]
    for model, decoder in choices.items():
        row = next(r for r in results if r["model"] == model and r["decoder"] == decoder)
        board.append(dict(pipeline=LABELS[model], decoder=decoder,
                          cv_macro_f1=float(cv[model, decoder]["macro_f1_all"]),
                          student_macro_f1=float(row["macro_f1"]), student_top1=float(row["top1"]),
                          reverse_errors=row["reverse_errors"], trips_at_090=row["trips_at_090"],
                          selected_by_cv=(model == winning["model"] and decoder == winning["decoder"])))
    csv_write(run / "comparison.csv", board)
    winner = next(r for r in board if r["selected_by_cv"])
    actual_wins = [r for r in per_trip if (r["model"], r["decoder"]) == key(winning)]
    csv_write(run / "selected_per_trip.csv", actual_wins)
    figure(run, board)

    best_test = max(board[2:], key=lambda r: r["student_macro_f1"])
    diff = winner["student_macro_f1"] - float(better["macro_f1"])
    report = [
        "# PathSnap integrated LightGBM experiment",
        "",
        f"The pipeline selected on the training folds was **{winner['pipeline']}** "
        f"with the `{winner['decoder']}` decoder. It scored **{winner['cv_macro_f1']:.4f}** "
        f"Macro F1 across 80 validation trips, then **{winner['student_macro_f1']:.4f} Macro F1** "
        f"and **{winner['student_top1']:.2%} Top-1** on the 10 student-test trips.",
        "",
        f"The best observed student-test result among fresh variants was **{best_test['student_macro_f1']:.4f}**. "
        f"The selected pipeline was {diff*100:+.2f} percentage points relative to the attached LightGBM replay. "
        f"{winner['trips_at_090']} of its 10 trips reached 0.90 Macro F1.",
        "",
        "| Pipeline | Decoder | CV Macro F1 | Student-test Macro F1 | Student-test Top-1 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in board:
        report.append(f"| {row['pipeline']} | `{row['decoder']}` | {row['cv_macro_f1']:.4f} | "
                      f"{row['student_macro_f1']:.4f} | {row['student_top1']:.2%} |")
    report += [
        "",
        "## Method",
        "",
        "- All variants ran on CPU, with the same trip splits, eight candidate arcs and existing aligned reference labels.",
        "- The 33-feature baseline combines the original 10 features with adjacent observations' candidate and road-connectivity context. "
        "The 51-feature variants add 18 multiscale direction features.",
        "- The filter omits trips `00000033` and `00000054` from fitting where enabled. All 80 training trips remain in the CV scoring population.",
        "- Decoder settings were selected on the training folds before student-test scoring. The grid tested the legacy topology, "
        "different emission weights, explicit reverse costs, direction-sensitive penalties, gap resets, graph distance and learned transition potentials.",
        "- The attached CV value is the reported all-trip figure. Its student-test value comes from the independent model replay. "
        "The student-test split has been used for development comparisons. Reserved evaluator trips were outside the experiment.",
        "",
        "[Full summary metrics](comparison.csv) · [Selected model by trip](selected_per_trip.csv) · [Comparison chart](macro_f1_comparison.png)",
    ]
    (run / "REPORT.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    print("\n".join(report[:6]))


def figure(run, rows):
    fig, ax = plt.subplots(figsize=(10.4, 4.5), layout="constrained")
    positions = np.arange(len(rows))[::-1]
    points = np.array([r["student_macro_f1"] for r in rows])
    ax.scatter(points, positions, s=74, color="#14766a")
    ax.axvline(.90, linestyle="--", color="#53615a", label="0.90 Macro F1 target")
    for point, y, row in zip(points, positions, rows):
        ax.annotate(f"{point:.4f}", (point, y), xytext=(8, 4), textcoords="offset points", fontsize=9)
    ax.set_yticks(positions, [r["pipeline"] for r in rows])
    ax.set_xlim(min(points)-.012, max(.910, max(points)+.015))
    ax.set_ylim(-.65, len(rows)-.35)
    ax.set_xlabel("Student-test Macro F1")
    ax.set_title("PathSnap integrated LightGBM experiment", loc="left", fontweight="bold", fontsize=15)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=.20)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, -.27), frameon=False)
    fig.savefig(run / "macro_f1_comparison.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "set1", "prepared", "archive"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    config, sweep, per_trip = verify(args.run, args.set1, args.prepared)
    write_report(args.run, args.archive, config, sweep, per_trip)
