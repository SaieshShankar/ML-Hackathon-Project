"""Server-side sources for rehearsal and organiser-supplied trips."""
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

import numpy as np


def normalized_id(text):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", text):
        raise ValueError(f"Invalid trip ID: {text}")
    return text


class TripSource:
    def __init__(self, mode, source, references=None):
        self.mode = mode
        self.source = source.resolve(strict=True)
        self.references = references.resolve(strict=True) if references else None
        self.is_zip = self.source.is_file()
        if mode == "rehearsal" and not self.is_zip:
            raise ValueError("Rehearsal expects the original Student folder.zip.")
        if self.references and not self.references.is_dir():
            raise ValueError("--references must be a directory.")
        if mode == "evaluation" and self.source.name.lower() in {"train", "student_test"}:
            raise ValueError("Evaluation mode requires the organiser's reserved dataset.")
        if self.is_zip:
            with ZipFile(self.source) as archive:
                names = set(archive.namelist())
            prefix = self._zip_prefixes(names)
            complete = lambda path: all(f"{path}.{ext}" in names for ext in ("track", "nodes", "arcs"))
        else:
            prefix = self._directory_prefixes()
            complete = lambda path: all(Path(f"{path}.{ext}").is_file() for ext in ("track", "nodes", "arcs"))
        if len(prefix) != 10:
            raise ValueError(f"Expected exactly 10 {mode} trips; found {len(prefix)}.")
        if any(not complete(path) for path in prefix.values()):
            raise ValueError("Every listed trip needs .track, .nodes and .arcs files.")
        self.trips = dict(sorted(prefix.items()))
        if mode == "evaluation":
            if self.references:
                missing = [trip for trip in self.trips if self._separate_reference(trip) is None]
            elif self.is_zip:
                missing = [trip for trip, path in self.trips.items() if f"{path}.route" not in names]
            else:
                missing = [trip for trip, path in self.trips.items()
                           if not Path(f"{path}.route").is_file()]
            if missing:
                raise ValueError("Evaluation requires one server-side .route file per trip. "
                                 "Use --references if the organisers supply them separately.")

    def _zip_prefixes(self, names):
        result = {}
        for name in sorted(names):
            parts = PurePosixPath(name).parts
            if not name.endswith(".track") or ".." in parts or "__MACOSX" in parts:
                continue
            if self.mode == "rehearsal" and "student_test" not in parts[:-1]:
                continue
            if self.mode == "evaluation" and set(parts[:-1]) & {"train", "student_test"}:
                continue
            trip = normalized_id(PurePosixPath(name).stem)
            if trip in result:
                raise ValueError(f"Duplicate trip ID: {trip}")
            result[trip] = name.removesuffix(".track")
        return result

    def _directory_prefixes(self):
        result = {}
        for path in sorted(self.source.rglob("*.track")):
            if set(path.relative_to(self.source).parts[:-1]) & {"train", "student_test"}:
                continue
            if not path.resolve().is_relative_to(self.source):
                continue
            trip = normalized_id(path.stem)
            if trip in result:
                raise ValueError(f"Duplicate trip ID: {trip}")
            result[trip] = path.with_suffix("")
        return result

    @contextmanager
    def _open(self, member):
        if self.is_zip:
            with ZipFile(self.source) as archive, archive.open(str(member)) as stream:
                yield stream
        else:
            member = Path(member).resolve(strict=True)
            if not member.is_relative_to(self.source):
                raise ValueError("Trip file lies outside the supplied directory.")
            with member.open("rb") as stream:
                yield stream

    def load_inputs(self, trip):
        prefix = self.trips[trip]
        result = []
        for ext, dtype in (("track", float), ("nodes", float), ("arcs", np.int32)):
            with self._open(f"{prefix}.{ext}") as stream:
                result.append(np.loadtxt(stream, dtype=dtype, ndmin=2))
        track, nodes, arcs = result
        if track.shape[1] != 3 or not track.size or nodes.shape[1] != 2 or not nodes.size:
            raise ValueError("Expected longitude/latitude/time in .track and longitude/latitude in .nodes.")
        if not np.isfinite(nodes).all() or (arcs.size and
                (arcs.shape[1] != 2 or arcs.min() < 0 or arcs.max() >= len(nodes))):
            raise ValueError("Road graph has invalid coordinates or node IDs.")
        return track, nodes, arcs

    def load_reference(self, trip):
        """Called only after prediction. Never include this data in an API response."""
        if self.references:
            path = self._separate_reference(trip)
            return np.loadtxt(path, dtype=np.int32, ndmin=1) if path else None
        try:
            with self._open(f"{self.trips[trip]}.route") as stream:
                return np.loadtxt(stream, dtype=np.int32, ndmin=1)
        except (KeyError, FileNotFoundError):
            return None

    def _separate_reference(self, trip):
        for path in (self.references/f"{trip}.route", self.references/trip/f"{trip}.route"):
            if path.is_file() and path.resolve().is_relative_to(self.references):
                return path
        return None
