"""Generate a self-contained Folium review demo from saved PathSnap predictions."""
import argparse
import base64
import csv
import json
import re
from pathlib import Path
from zipfile import ZipFile

import folium
import numpy as np
from branca.element import Element, MacroElement
from jinja2 import Template
from scipy.spatial import cKDTree

MODEL = "Histogram Gradient Boosting"
GPS, GRAPH, REFERENCE, PREDICTION, BASELINE = "#2463ae", "#84929a", "#912453", "#087b70", "#9a5610"


def offline_assets(map_, vendor):
    """Inline Folium's Leaflet and jQuery dependencies and image assets."""
    figure = map_.get_root()
    map_.default_js = []
    map_.default_css = []
    for name in ("leaflet.js", "jquery.min.js"):
        script = (vendor / name).read_text(encoding="utf-8")
        script = re.sub(r"(?m)^//[#@] sourceMappingURL=.*$", "", script)
        figure.header.add_child(Element("<script>" + script + "</script>"), name)
    style = (vendor / "leaflet.css").read_text(encoding="utf-8")

    def embed(match):
        binary = (vendor / match[1]).read_bytes()
        return "url(data:image/png;base64," + base64.b64encode(binary).decode("ascii") + ")"

    style = re.sub(r"url\(images/([^)]+)\)", embed, style)
    figure.header.add_child(Element("<style>" + style + "</style>"), "offline-leaflet-css")
    for name in ("LICENSE", "jquery-LICENSE"):
        text = (vendor / name).read_text(encoding="utf-8")
        figure.header.add_child(Element("<!--\n" + text + "\n-->"), name)


def line_layer(map_, nodes, arcs, ids, name, color, weight, pane, show=True, dash=None):
    group = folium.FeatureGroup(name=name, show=show).add_to(map_)
    # Draw each physical geometry once. Arc direction remains visible in example popups.
    endpoints = np.unique(np.sort(arcs[np.unique(ids)], axis=1), axis=0)
    coordinates = nodes[endpoints][:, :, ::-1].round(7).tolist()
    folium.PolyLine(
        coordinates, color=color, weight=weight, opacity=.86 if weight > 2 else .46,
        dash_array=dash, pane=pane, interactive=False, smooth_factor=.6,
    ).add_to(group)


def find_examples(track, selected, reference, nearest, candidates):
    recovered = np.flatnonzero((selected == reference) & (nearest != reference))
    residual = np.flatnonzero((selected != reference) & (nearest == reference)
                              & np.any(candidates == reference[:, None], axis=1))
    first = int(recovered[len(recovered) // 2])
    distances = np.linalg.norm(track[residual, :2] - track[first, :2], axis=1)
    second = int(residual[distances.argmax()])
    return first, second


def add_example(group, track, arcs, reference, selected, nearest, scores, candidates,
                index, title, color, helpful):
    ref, pred, base = (int(x[index]) for x in (reference, selected, nearest))
    details = (
        "The nearest segment and aligned reference have different arc IDs."
        if helpful else
        "The reference was among the eight candidates, but boosting preferred another arc."
    )
    if helpful and np.array_equal(arcs[base], arcs[ref][::-1]):
        details = "The nearest segment is the same stretch of road in the opposite direction."

    def describe(arc):
        start, end = arcs[arc]
        return f"arc {arc} ({start} &rarr; {end})"

    score = float(scores[index, candidates[index] == pred][0])
    markup = (
        "<div class='example-popup'>"
        f"<strong>{title}</strong><p>GPS observation #{index+1:,}</p>"
        f"<b>Aligned reference:</b> {describe(ref)}<br>"
        f"<b>Boosting:</b> {describe(pred)}<br>"
        f"<b>Nearest segment:</b> {describe(base)}<br>"
        f"<b>Boosting candidate score:</b> {score:.3f}<p>{details}</p>"
        "</div>"
    )
    location = track[index, [1, 0]].tolist()
    return folium.CircleMarker(
        location, radius=9, color="white", weight=2, fill=True, fill_color=color,
        fill_opacity=1, pane="examplesPane", tooltip=title,
        popup=folium.Popup(markup, max_width=300),
    ).add_to(group)


def add_review_panel(map_, trip_id, count, model_accuracy, baseline_accuracy, examples, bounds):
    controls = MacroElement()
    controls._template = Template("""
    {% macro html(this, kwargs) %}
    <section class="review-panel" aria-label="PathSnap demo">
      <div class="review-label">PATHSNAP</div>
      <h1>Trip {{ this.trip_id }}</h1>
      <div class="review-meta">Student test · {{ this.count }} GPS observations</div>
      <div class="review-scores">
        <div><strong>{{ this.model_accuracy }}</strong><span>Gradient Boosting</span></div>
        <div><strong>{{ this.baseline_accuracy }}</strong><span>Nearest segment</span></div>
      </div>
      <div class="review-note">Top-1 against the aligned reference for this trip.</div>
      <div class="review-buttons">
        <button id="corrected-example">Show correction</button>
        <button id="remaining-error">Show error</button>
        <button id="full-trip">Full trip</button>
      </div>
    </section>
    {% endmacro %}
    {% macro script(this, kwargs) %}
    const demoMap = {{ this._parent.get_name() }};
    const exampleGroup = {{ this.group_name }};
    function focusExample(marker) {
      if (!demoMap.hasLayer(exampleGroup)) demoMap.addLayer(exampleGroup);
      demoMap.setView(marker.getLatLng(), 17);
      marker.openPopup();
    }
    document.getElementById('corrected-example').onclick = () => focusExample({{ this.corrected }});
    document.getElementById('remaining-error').onclick = () => focusExample({{ this.remaining }});
    document.getElementById('full-trip').onclick = () => {
      demoMap.closePopup();
      demoMap.fitBounds({{ this.bounds }}, {padding: [60, 60]});
    };
    {% endmacro %}
    """)
    controls.trip_id, controls.count = trip_id, f"{count:,}"
    controls.model_accuracy, controls.baseline_accuracy = f"{model_accuracy:.2%}", f"{baseline_accuracy:.2%}"
    controls.group_name = examples[0]._parent.get_name()
    controls.corrected, controls.remaining = (example.get_name() for example in examples)
    controls.bounds = json.dumps(bounds)
    controls.add_to(map_)


STYLE = """
<style>
.leaflet-container { background: #f2f4f2; font-family: system-ui, sans-serif; }
.review-panel {
  position: fixed; top: 12px; left: 52px; z-index: 1000; width: 300px;
  box-sizing: border-box; padding: 15px 16px 14px; color: #172e2b;
  background: #fffffff7; border: 1px solid #dce4df; border-radius: 12px;
  box-shadow: 0 8px 28px #17362b20; font: 13px/1.4 system-ui, sans-serif;
}
.review-panel h1 { margin: 3px 0 0; font-size: 23px; line-height: 1.2; }
.review-label { color: #087b70; font-weight: 750; font-size: 11px; letter-spacing: .1em; }
.review-meta,.review-note { color: #465b55; margin-top: 4px; font-size: 11.5px; }
.review-scores { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 12px 0 2px; }
.review-scores div { background: #f0f6f3; border-radius: 6px; padding: 7px 8px; }
.review-scores strong,.review-scores span { display: block; }
.review-scores strong { font-size: 19px; line-height: 1.15; }
.review-scores span { font-size: 11px; }
.review-buttons { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; }
.review-buttons button {
  cursor: pointer; border: 1px solid #b4c8be; background: white; color: #21443a;
  padding: 5px 7px; border-radius: 5px; font: 600 11px system-ui,sans-serif;
}
.review-buttons button:hover,.review-buttons button:focus-visible { background: #e6f2ed; }
.layer-key {
  display: inline-block; width: 15px; border-top: 3px solid currentColor;
  vertical-align: middle; margin-right: 5px;
}
.example-popup { width: min(250px, 55vw); line-height: 1.5; }
.example-popup p { margin: 6px 0; }
@media (max-width: 650px) {
  .review-panel { top: auto; bottom: 17px; left: 10px; width: min(300px,calc(100vw - 25px)); }
}
</style>
"""


def layer_label(color, text, dashed=False):
    border = "border-top-style:dashed;" if dashed else ""
    return f'<span class="layer-key" style="color:{color};{border}"></span>{text}'


def build_demo(dataset, evidence, vendor, trip_id, output):
    with ZipFile(dataset) as archive:
        folder = f"Student folder/student_test/{trip_id}/{trip_id}"
        def load(suffix, dtype):
            with archive.open(f"{folder}.{suffix}") as handle:
                return np.loadtxt(handle, dtype=dtype)
        track, nodes, arcs, route = (load("track", float), load("nodes", float),
                                     load("arcs", int), load("route", int))
    with np.load(evidence / "predictions" / f"histogram_gradient_boosting_student_test_{trip_id}.npz") as saved:
        selected, reference, candidates, scores = (
            saved[key] for key in ("selected", "reference", "candidate_arcs", "scores"))
    nearest = candidates[:, 0]
    assert len(selected) == len(track) == len(reference)
    assert np.array_equal(candidates[np.arange(len(track)), scores.argmax(axis=1)], selected)

    model_accuracy = np.mean(selected == reference)
    baseline_accuracy = np.mean(nearest == reference)
    with (evidence / "per_trip_metrics.csv").open(encoding="utf-8", newline="") as file:
        metrics = {r["model"]: r for r in csv.DictReader(file)
                   if r["split"] == "student_test" and r["trip_id"] == trip_id}
    assert np.isclose(model_accuracy, float(metrics[MODEL]["top1_accuracy"]))
    assert np.isclose(baseline_accuracy, float(metrics["Nearest segment (heuristic)"]["top1_accuracy"]))

    coordinates = track[:, [1, 0]]
    bounds = [coordinates.min(axis=0).tolist(), coordinates.max(axis=0).tolist()]
    map_ = folium.Map(location=coordinates.mean(axis=0).tolist(), tiles=None,
                      prefer_canvas=True, max_zoom=20, zoom_snap=.25, fade_animation=False)
    offline_assets(map_, vendor)
    map_.get_root().header.add_child(Element("<title>PathSnap | Trip " + trip_id + "</title>" + STYLE))
    for pane, z in (("roadsPane", 410), ("baselinePane", 420), ("predictionPane", 430),
                    ("referencePane", 440), ("gpsPane", 445), ("examplesPane", 455)):
        folium.map.CustomPane(pane, z_index=z, pointer_events=pane == "examplesPane").add_to(map_)
    origin = track[:, :2].mean(axis=0)
    scale = np.pi * 6_371_008.8 / 180 * np.array([np.cos(np.deg2rad(origin[1])), 1])
    track_tree = cKDTree((track[:, :2] - origin) * scale)
    midpoints = nodes[arcs].mean(axis=1)
    midpoint_distance, _ = track_tree.query((midpoints - origin) * scale)
    corridor = np.union1d(np.flatnonzero(midpoint_distance <= 450), candidates.ravel())
    line_layer(map_, nodes, arcs, corridor, layer_label(GRAPH, "Nearby road network"),
               GRAPH, 1.5, "roadsPane")
    line_layer(map_, nodes, arcs, nearest, layer_label(BASELINE, "Nearest-segment baseline"),
               BASELINE, 4, "baselinePane", show=False)
    line_layer(map_, nodes, arcs, selected, layer_label(PREDICTION, "Gradient Boosting predictions"),
               PREDICTION, 5, "predictionPane")
    line_layer(map_, nodes, arcs, route, layer_label(REFERENCE, "Supplied reference route", True),
               REFERENCE, 3, "referencePane", dash="7 6")
    gps = folium.FeatureGroup(name=layer_label(GPS, "Raw GPS observations")).add_to(map_)
    for location in coordinates.round(7):
        folium.CircleMarker(
            location.tolist(), radius=1.7, color=GPS, weight=0, fill=True,
            fill_opacity=.60, pane="gpsPane", interactive=False,
        ).add_to(gps)
    group = folium.FeatureGroup(name="Review examples (2 markers)").add_to(map_)
    first, second = find_examples(track, selected, reference, nearest, candidates)
    markers = [
        add_example(group, track, arcs, reference, selected, nearest, scores, candidates,
                    first, "Boosting corrects nearest-road choice", PREDICTION, True),
        add_example(group, track, arcs, reference, selected, nearest, scores, candidates,
                    second, "Boosting still chooses the wrong arc", BASELINE, False),
    ]
    map_.fit_bounds(bounds, padding=(60, 60))
    folium.LayerControl(collapsed=False).add_to(map_)
    add_review_panel(map_, trip_id, len(track), model_accuracy, baseline_accuracy, markers, bounds)
    map_.save(output)
    print(f"Built offline Folium map for student-test trip {trip_id}.")
    print(f"GPS points={len(track)}, graph nodes={len(nodes)}, arcs={len(arcs)}, route entries={len(route)}")
    print(f"Correction at GPS observation {first+1}, remaining error at {second+1}.")
    print(f"Boosting accuracy={model_accuracy:.6f}, nearest accuracy={baseline_accuracy:.6f}.")
    for i in (first, second):
        print(f"Observation {i+1}: aligned={reference[i]}, boosting={selected[i]}, nearest={nearest[i]}")
    print(f"Output={output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--trip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_demo(args.dataset, args.evidence, args.vendor, args.trip, args.output)
