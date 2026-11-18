"""Fetch and verify the assets embedded by the offline PathSnap map."""
import argparse
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

LEAFLET = "https://cdn.jsdelivr.net/npm/leaflet@1.9.3/"
JQUERY = "https://cdn.jsdelivr.net/npm/jquery@3.7.1/"
ASSETS = {
    "LICENSE": (LEAFLET + "LICENSE", "97c194191510b19c72d25094922821a888d5d9bafb909ea7a38c9bf5b6c40a08"),
    "leaflet.js": (LEAFLET + "dist/leaflet.js", "5819285cec137b229c94e1ee5ad73e8b6b84345a4367d60f75fe477fe0fb7b03"),
    "leaflet.css": (LEAFLET + "dist/leaflet.css", "90b693d86392a4779c861b28cf307e7e59c3fb35328c4d8b95f58f814d38c722"),
    "layers.png": (LEAFLET + "dist/images/layers.png", "1dbbe9d028e292f36fcba8f8b3a28d5e8932754fc2215b9ac69e4cdecf5107c6"),
    "layers-2x.png": (LEAFLET + "dist/images/layers-2x.png", "066daca850d8ffbef007af00b06eac0015728dee279c51f3cb6c716df7c42edf"),
    "marker-icon.png": (LEAFLET + "dist/images/marker-icon.png", "574c3a5cca85f4114085b6841596d62f00d7c892c7b03f28cbfa301deb1dc437"),
    "marker-icon-2x.png": (LEAFLET + "dist/images/marker-icon-2x.png", "00179c4c1ee830d3a108412ae0d294f55776cfeb085c60129a39aa6fc4ae2528"),
    "marker-shadow.png": (LEAFLET + "dist/images/marker-shadow.png", "264f5c640339f042dd729062cfc04c17f8ea0f29882b538e3848ed8f10edb4da"),
    "jquery.min.js": (JQUERY + "dist/jquery.min.js", "fc9a93dd241f6b045cbff0481cf4e1901becd0e12fb45166a8f17f95823f0b1a"),
    "jquery-LICENSE": (JQUERY + "LICENSE.txt", "d4db9ebe6f29f5168eac45ad713f055623ac5d0dcd5ba92da23d650ae012020d"),
}


def fetch_assets(destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name, (url, expected_digest) in ASSETS.items():
        file = destination / name
        if file.is_file() and sha256(file.read_bytes()).hexdigest() == expected_digest:
            continue
        with urlopen(url, timeout=30) as response:
            data = response.read()
        if sha256(data).hexdigest() != expected_digest:
            raise ValueError(f"Unexpected SHA-256 for {name}")
        file.write_bytes(data)
    print(f"Verified {len(ASSETS)} map assets in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("leaflet"))
    fetch_assets(parser.parse_args().output)
