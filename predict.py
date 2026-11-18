"""Run the champion on one trip using its GPS, node and arc files."""
import argparse
from pathlib import Path

import numpy as np

from pathsnap.io import write_rows
from pathsnap.demo_service import predict_track
from pathsnap.phase2 import load_decoder
from pathsnap.pipeline import default_model_dir, load_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("track", "nodes", "arcs"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", type=Path, default=default_model_dir())
    parser.add_argument("--output", type=Path, default=Path("predicted_arcs.csv"))
    args = parser.parse_args()
    track = np.loadtxt(args.track, dtype=float, ndmin=2)
    if not track.size:
        parser.error("The track has no GPS observations.")
    nodes = np.loadtxt(args.nodes, dtype=float, ndmin=2)
    arcs = np.loadtxt(args.arcs, dtype=np.int32, ndmin=2)
    predictions, status, _ = predict_track(track, nodes, arcs, load_models(args.model), load_decoder(args.model))
    args.output.parent.mkdir(exist_ok=True, parents=True)
    write_rows(args.output, [
        dict(observation_index=i, timestamp=track[i, 2], predicted_arc_id=int(arc),
             path_status=status[i])
        for i, arc in enumerate(predictions)])
    print(f"Saved {len(predictions):,} arc predictions to {args.output}.")


if __name__ == "__main__":
    main()
