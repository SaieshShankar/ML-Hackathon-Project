"""Read the student archive and prepared trip features."""
import csv
from pathlib import PurePosixPath

import numpy as np


def archive_trips(archive):
    trips = {"train": {}, "student_test": {}}
    for name in sorted(archive.namelist()):
        parts = PurePosixPath(name).parts
        if len(parts) == 4 and parts[1] in trips and parts[3] == f"{parts[2]}.track":
            trips[parts[1]][parts[2]] = "/".join(parts[:3]) + "/" + parts[2]
    if [len(split) for split in trips.values()] != [80, 10]:
        raise ValueError("Expected 80 training trips and 10 student-test trips.")
    if set(trips["train"]) & set(trips["student_test"]):
        raise ValueError("Training and student-test trip IDs overlap.")
    return trips


def zip_trip(archive, prefix):
    files = [("track", float, 2), ("nodes", float, 2), ("arcs", np.int32, 2),
             ("route", np.int32, 1)]
    arrays = []
    for extension, dtype, dimensions in files:
        with archive.open(f"{prefix}.{extension}") as handle:
            arrays.append(np.loadtxt(handle, dtype=dtype, ndmin=dimensions))
    return arrays


def load_prepared(path):
    with np.load(path) as archive:
        return {k: archive[k] for k in archive.files}


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
