"""Evaluate the saved champion on prepared student-test trips."""
import argparse
import json
from pathlib import Path

import numpy as np

from pathsnap.io import load_prepared, write_rows
from pathsnap.metrics import evaluate
from pathsnap.phase2 import load_decoder, predict
from pathsnap.pipeline import MODEL_DIR, default_model_dir, load_models, predict_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("prepared"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluation"))
    parser.add_argument("--legacy", action="store_true",
                        help="Use the original Phase 1 decoder for its frozen baseline.")
    args = parser.parse_args()
    trips = sorted((args.prepared / "student_test").glob("*.npz"))
    if not trips:
        parser.error("No prepared student-test trips found.")
    directory = args.model or (MODEL_DIR if args.legacy else default_model_dir())
    models = load_models(directory)
    settings = load_decoder(directory)
    rows = []
    for path in trips:
        data = load_prepared(path)
        if not args.legacy and "elapsed" not in data:
            parser.error("Prepared trips need timestamps for Phase 2. Re-run prepare.py.")
        selected = predict_indices(data, *models) if args.legacy else predict(data, *models, settings)
        _, stats = evaluate(selected, data["candidates"], data["reference"], data["route"],
                            data["arcs"], data["finite"], data["reset"])
        rows.append(dict(trip_id=path.stem, **stats))
    summary = dict(trips=len(rows), macro_f1=float(np.mean([r["macro_f1"] for r in rows])),
                   top1=sum(r["correct"] for r in rows) / sum(r["observations"] for r in rows))
    args.output.mkdir(exist_ok=True, parents=True)
    write_rows(args.output / "per_trip.csv", rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
