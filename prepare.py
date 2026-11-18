"""Prepare the champion's features and labels from the original student ZIP."""
import argparse
from hashlib import file_digest
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from pathsnap.io import archive_trips, zip_trip
from pathsnap.pipeline import preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("prepared"))
    parser.add_argument("--split", choices=("train", "student_test"),
                        help="Prepare just one partition.")
    parser.add_argument("--only-trip", help="Prepare one trip for a quick check.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.dataset.open("rb") as handle:
        digest = file_digest(handle, "sha256").hexdigest()
    with ZipFile(args.dataset) as archive:
        trips = archive_trips(archive)
        if args.split:
            trips = {args.split: trips[args.split]}
        if args.only_trip and not any(args.only_trip in split for split in trips.values()):
            parser.error("Trip ID not found in either student partition.")
        for split, trip_prefixes in trips.items():
            target = args.output / split
            target.mkdir(exist_ok=True)
            for trip, prefix in trip_prefixes.items():
                if args.only_trip and trip != args.only_trip:
                    continue
                data = preprocess(*zip_trip(archive, prefix))
                np.savez(target / f"{trip}.npz", **data)
                print(f"Prepared {split}/{trip} ({len(data['reference']):,} GPS points)", flush=True)
    (args.output / "dataset_sha256.txt").write_text(digest + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
