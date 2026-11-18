"""Summarise and verify the saved PathSnap benchmark evidence."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss


def verify(path, trips, assignments):
    split = assignments.groupby("fold")
    assert len(split) == 5
    assert (split.size() == 80).all()
    assert assignments.query("role == 'validation'").trip_id.nunique() == 80
    assert not assignments.duplicated(["fold", "trip_id"]).any()
    assert (trips.groupby(["split", "model"]).size().unstack() ==
            np.array([[80] * 6, [10] * 6])).all().all()
    assert not trips.duplicated(["split", "model", "trip_id"]).any()
    assert set(trips.query("split == 'cv'").trip_id).isdisjoint(trips.query("split == 'student_test'").trip_id)
    cv = trips.query("split == 'cv'")[["fold", "trip_id"]].drop_duplicates()
    assert len(cv.merge(assignments.query("role == 'validation'"), on=["fold", "trip_id"])) == 80
    recoveries, recoverable, regressions, baseline_correct = [], [], [], []
    for row in trips.itertuples():
        key = row.model.lower().replace(" ", "_")
        file = path / "predictions" / f"{key}_{row.split}_{row.trip_id}.npz"
        with np.load(file) as data:
            selected, reference, arcs, scores = (data[x] for x in ("selected", "reference", "candidate_arcs", "scores"))
            assert len(selected) == len(reference) == row.observations
            assert np.isclose(np.mean(selected == reference), row.top1_accuracy)
            assert np.isclose(f1_score(reference, selected, average="macro", zero_division=0), row.observation_macro_f1)
            target = arcs == reference[:, None]
            assert np.isclose(np.mean(~target.any(axis=1)), row.candidate_miss_rate)
            nearest_correct = arcs[:, 0] == reference
            can_recover = ~nearest_correct & target.any(axis=1)
            correct = selected == reference
            recoveries.append(np.count_nonzero(can_recover & correct))
            recoverable.append(np.count_nonzero(can_recover))
            regressions.append(np.count_nonzero(nearest_correct & ~correct))
            baseline_correct.append(np.count_nonzero(nearest_correct))
            if scores.size:
                assert np.array_equal(arcs[np.arange(len(arcs)), np.argmax(scores, axis=1)], selected)
                assert np.isclose(log_loss(target.ravel(), scores.ravel(), labels=[False, True]),
                                  row.candidate_pair_log_loss)
    trips["nearest_error_recovered"] = recoveries
    trips["nearest_error_recoverable"] = recoverable
    trips["nearest_correct_changed_to_wrong"] = regressions
    trips["nearest_correct"] = baseline_correct
    print(f"Verified {len(trips)} prediction archives against the recorded metrics.", flush=True)


def summarize(path, trips):
    times = pd.read_csv(path / "fit_timings.csv")
    rows = []
    for model, scores in trips.groupby("model"):
        cv = scores.query("split == 'cv'")
        fold_f1 = cv.groupby("fold").observation_macro_f1.mean()
        t = scores.query("split == 'student_test'")
        fit = times[(times.model == model) & (times.split == "student_test")].iloc[0]
        rows.append({
            "model": model,
            "cv_macro_f1": fold_f1.mean(),
            "cv_macro_f1_fold_sd": fold_f1.std(),
            "cv_top1_trip_mean": cv.top1_accuracy.mean(),
            "student_macro_f1": t.observation_macro_f1.mean(),
            "student_top1_micro": np.average(t.top1_accuracy, weights=t.observations),
            "student_top1_trip_mean": t.top1_accuracy.mean(),
            "student_candidate_precision_trip_mean": t.candidate_pair_precision.mean(),
            "student_candidate_recall_trip_mean": t.candidate_pair_recall.mean(),
            "student_candidate_f1_trip_mean": t.candidate_pair_f1.mean(),
            "student_candidate_log_loss": np.average(t.candidate_pair_log_loss, weights=t.observations),
            "student_candidate_miss_rate": np.average(t.candidate_miss_rate, weights=t.observations),
            "student_recovered_nearest_errors": t.nearest_error_recovered.sum(),
            "student_recoverable_nearest_errors": t.nearest_error_recoverable.sum(),
            "student_regressed_nearest_correct": t.nearest_correct_changed_to_wrong.sum(),
            "student_direct_continuity_trip_mean": t.direct_continuity.mean(),
            "student_median_scoring_ms": t.scoring_ms.median(),
            "student_max_scoring_ms": t.scoring_ms.max(),
            "final_fit_seconds": fit.fit_seconds,
        })
    summary = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    summary.to_csv(path / "comparison.csv", index=False, float_format="%.8f")
    fold_metrics = trips.query("split == 'cv'").groupby(["model", "fold"])[
        ["observation_macro_f1", "top1_accuracy", "candidate_pair_log_loss", "direct_continuity"]
    ].mean()
    fold_metrics.to_csv(path / "cv_fold_metrics.csv", float_format="%.8f")
    return summary


def plot(path, summary):
    models = summary.query("~model.str.startswith('Nearest')", engine="python").iloc[::-1]
    positions = np.arange(len(models))
    lower = 100 * (models.cv_macro_f1 - models.cv_macro_f1_fold_sd).min() - 1
    upper = 100 * max((models.cv_macro_f1 + models.cv_macro_f1_fold_sd).max(),
                      models.student_macro_f1.max()) + 2
    fig, axes = plt.subplots(
        1, 2, figsize=(12.4, 4.6), sharex=True, sharey=True, layout="constrained")
    for ax, col, title in (
        (axes[0], "cv_macro_f1", "Training cross-validation (±1 fold SD)"),
        (axes[1], "student_macro_f1", "Student test"),
    ):
        values = models[col] * 100
        ax.scatter(values, positions, color="#176c66", s=68, zorder=3)
        if col == "cv_macro_f1":
            ax.errorbar(values, positions, xerr=models.cv_macro_f1_fold_sd * 100,
                        color="#176c66", fmt="none", capsize=4)
        for x, y in zip(values, positions):
            ax.annotate(f"{x:.2f}%", (x, y), xytext=(6, 9), textcoords="offset points", fontsize=9)
        ax.set_yticks(positions, labels=models.model)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
        ax.set_xlabel("Observation-level macro F1 (%)")
        ax.set_ylim(-.55, len(positions) - .35)
        ax.grid(axis="x", color="#e2e8e6")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(lower, upper)
    fig.suptitle("PathSnap candidate classifier comparison", x=.02, ha="left", fontweight="bold", fontsize=17)
    fig.savefig(path / "model_comparison.png", dpi=170)
    plt.close(fig)

    loss = pd.read_csv(path / "mlp_training_loss.csv")
    fig, axs = plt.subplots(figsize=(8, 4), layout="constrained")
    for fold, rows in loss.groupby("fold"):
        label = "Full training fit" if fold == 0 else f"CV fold {fold}"
        axs.plot(rows.epoch, rows.training_loss, label=label)
    axs.set(title="Shallow MLP training loss", xlabel="Epoch", ylabel="Training loss")
    axs.legend(frameon=False)
    fig.savefig(path / "mlp_training_loss.png", dpi=160)
    plt.close(fig)


def main():
    path = Path(sys.argv[1])
    trips = pd.read_csv(path / "per_trip_metrics.csv", dtype={"trip_id": str})
    assignments = pd.read_csv(path / "fold_assignments.csv", dtype={"trip_id": str})
    verify(path, trips, assignments)
    summary = summarize(path, trips)
    plot(path, summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
