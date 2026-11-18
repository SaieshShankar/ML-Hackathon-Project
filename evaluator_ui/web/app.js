"use strict";

const $ = (id) => document.getElementById(id);
const map = L.map("map", { preferCanvas: true, zoomSnap: 0.25, maxZoom: 20 }).setView([0, 0], 2);
const groups = {
  roads: L.layerGroup().addTo(map),
  gps: L.layerGroup().addTo(map),
  route: L.layerGroup().addTo(map)
};
map.createPane("road").style.zIndex = 410;
map.createPane("points").style.zIndex = 420;
map.createPane("match").style.zIndex = 430;
L.control.scale({ imperial: false, maxWidth: 110 }).addTo(map);
let bounds;

function connection(state, message) {
  $("connection").dataset.state = state;
  $("connection").querySelector("strong").textContent = message;
}
async function json(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : "The request was rejected by Laptop 1.");
  }
  return data;
}
function error(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}
async function listTrips() {
  error("");
  connection("pending", "Connecting");
  $("reconnect").hidden = true;
  $("trip").disabled = true;
  $("run").disabled = true;
  try {
    const config = await json("/client-config.json");
    $("backend").textContent = config.backend_url;
    const [health, data] = await Promise.all([json("/api/health"), json("/api/trips")]);
    if (health.status !== "ready" || !Array.isArray(data.trip_ids)) {
      throw new Error("Laptop 1 is still starting.");
    }
    $("dataset-mode").textContent = data.mode === "evaluation"
      ? "Reserved evaluator trips · " + data.trip_ids.length + " available"
      : "Rehearsal · " + data.trip_ids.length + " student-test trips";
    $("trip").replaceChildren(new Option("Choose a trip", ""));
    data.trip_ids.forEach(id => $("trip").add(new Option(id, id)));
    $("trip").disabled = false;
    connection("ready", "Laptop 1 connected");
  } catch (problem) {
    error(problem.message);
    $("dataset-mode").textContent = "Connect to Laptop 1 to load the trip list.";
    connection("error", "Laptop 1 unavailable");
    $("reconnect").hidden = false;
  }
}
function fit() {
  if (bounds && bounds.isValid()) map.fitBounds(bounds, { padding: [24, 24], maxZoom: 17, animate: false });
}
function percent(value) {
  return typeof value === "number" ? (100 * value).toFixed(2) + "%" : "—";
}
function draw(result) {
  Object.values(groups).forEach(layer => layer.clearLayers());
  const display = result.map;
  const roads = L.polyline(display.roads, { pane: "road", color: "#81948b", weight: 1.4, opacity: .58, interactive: false });
  const route = L.polyline(display.route, { pane: "match", color: "#087c65", weight: 4, opacity: .92, interactive: false });
  roads.addTo(groups.roads);
  route.addTo(groups.route);
  for (const point of display.gps) {
    L.circleMarker(point, { pane: "points", radius: 2.1, stroke: false,
                           fillColor: "#2165a5", fillOpacity: .6, interactive: false }).addTo(groups.gps);
  }
  bounds = L.latLngBounds(display.gps);
  if (route.getBounds().isValid()) bounds.extend(route.getBounds());
  $("map-empty").hidden = bounds.isValid();
  $("layers").disabled = !bounds.isValid();
  $("fit").disabled = !bounds.isValid();
  map.invalidateSize();
  fit();

  $("macro-f1").textContent = percent(result.metrics?.macro_f1);
  $("top1").textContent = percent(result.metrics?.top1);
  $("observations").textContent = result.matched_observations.toLocaleString() + " / " + result.observations.toLocaleString();
  $("latency").textContent = Math.round(result.inference_ms).toLocaleString() + " ms";
  $("result-title").textContent = "Predicted route · " + result.trip_id;
  const breaks = (result.breaks.gps_gap || 0) + (result.breaks.disconnected || 0) + (result.breaks.after_missing || 0);
  $("result-sub").textContent = result.metrics
    ? result.model + " · " + breaks + (breaks === 1 ? " route break" : " route breaks")
    : result.model + " · Server-side scores unavailable";
  $("request").hidden = false;
  $("request-id").textContent = result.request_id;
}
$("trip").addEventListener("change", () => { $("run").disabled = !$("trip").value; });
$("reconnect").addEventListener("click", listTrips);
$("fit").addEventListener("click", fit);
for (const name of Object.keys(groups)) {
  $("layer-" + name).addEventListener("change", (event) => {
    if (event.target.checked) groups[name].addTo(map);
    else groups[name].remove();
  });
}
$("trip-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const trip = $("trip").value;
  if (!trip) return;
  $("trip").disabled = true;
  $("run").disabled = true;
  $("live").textContent = "Running on Laptop 1…";
  error("");
  try {
    const prediction = await json("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trip_id: trip })
    });
    draw(prediction);
    $("live").textContent = "Live prediction complete";
  } catch (problem) {
    error(problem.message);
    $("live").textContent = "";
  } finally {
    $("trip").disabled = false;
    $("run").disabled = false;
  }
});
listTrips();
