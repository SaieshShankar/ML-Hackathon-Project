# PathSnap two-laptop demo

## What goes where

| Laptop | Runs | Receives |
|---|---|---|
| **1** | FastAPI, Phase 2 LightGBM, Viterbi and scoring | The PathSnap project with model files, road graphs, GPS and server-side reference routes |
| **2** | Browser UI and a small Python relay | Only **PathSnap_Laptop2_UI.zip** and Laptop 1's current LAN URL |

Both laptops need to be able to reach each other on a trusted Wi-Fi network or hotspot. Laptop 2 needs Python 3 and a browser. Once the Python packages for Laptop 1 are installed, the demo runs without internet access. The map draws the supplied road geometry and uses bundled Leaflet assets.

## Start Laptop 1

Open PowerShell in the PathSnap project. This checkout already has a Python environment. Reuse it if its interpreter is present. On a fresh checkout, create it once and install the dependencies.

    if (-not (Test-Path -LiteralPath ".\.venv\Scripts\python.exe")) {
        python -m venv .venv
        .\.venv\Scripts\python.exe -m pip install -r requirements-api.txt
    }

If Python reports a missing package when starting the API, run `.\.venv\Scripts\python.exe -m pip install -r requirements-api.txt` using the existing environment.

For rehearsal, start the API with the original student ZIP. It will show exactly the 10 student-test trips.

    .\.venv\Scripts\python.exe serve_api.py --mode rehearsal --data "C:\path\to\Student folder.zip" --host 0.0.0.0 --port 8000

Use `ipconfig` to find Laptop 1's active Wi-Fi IPv4 address. Send your teammate the UI ZIP and a URL such as `http://192.168.1.12:8000`, substituting the actual address. Keep the API window open for the live logs. In a second window, you can check `http://127.0.0.1:8000/api/health`.

If you are using the background rehearsal started in this checkout, you can follow its incoming requests from the project folder:

    Get-Content ".\Old Models\Local working files\demo-validation\active-rehearsal.stderr.log" -Tail 12 -Wait

Before restarting the backend for the final demo, stop that rehearsal process. Its PID is shown in the log line beginning `Started server process`. Use Task Manager's **Details** tab to end that identified Python process.

### Switch to the reserved evaluator trips

Once the organisers provide their data, stop rehearsal with Ctrl+C. Use evaluation mode with the reserved ZIP or extracted trip folder.

    .\.venv\Scripts\python.exe serve_api.py --mode evaluation --data "C:\path\to\Evaluator trips.zip" --host 0.0.0.0 --port 8000

The adapter expects 10 unique trip IDs. Each ID's files must use the same stem, such as `00000001.track`, `00000001.nodes`, `00000001.arcs`, and `00000001.route`. Trips can be nested in the archive or folder. If the organisers supply reference routes separately, add `--references "C:\path\to\Evaluator routes"`. The reference directory accepts one `TRIP_ID.route` file per trip at the root or within a matching trip subfolder. Confirm the organisers' handoff format and scoring protocol before demo day.

Laptop 1 reads the reference only after inference. It aligns the ordered reference arcs to the GPS observations to calculate per-trip Top-1 and Macro F1. Laptop 2 receives prediction geometry, GPS points, nearby roads and aggregate results. Keep the evaluator files on Laptop 1.

## Start Laptop 2

1. Extract **PathSnap_Laptop2_UI.zip**.
2. On Windows, double-click **START_UI.bat** and enter Laptop 1's URL when prompted. For a terminal launch on any platform, use `python serve_ui.py --api http://LAPTOP1-IP:8000 --open`.
3. The browser opens at `http://127.0.0.1:8765/` on Laptop 2. Wait for **Laptop 1 connected**, then check that the trip menu shows the right mode and ten trip IDs.

If the UI shows **Laptop 1 unavailable**, test from Laptop 2's PowerShell:

    Test-NetConnection LAPTOP1-IP -Port 8000
    Invoke-RestMethod http://LAPTOP1-IP:8000/api/health -TimeoutSec 5

If the health endpoint returns `status: ready`, restart the UI launcher with that same API URL. If the TCP check fails, confirm the health endpoint works on Laptop 1 first. When Laptop 1 is healthy, put both laptops on the same trusted hotspot to rule out Wi-Fi client isolation. Run `ipconfig` again on Laptop 1 after switching networks. Restart the Laptop 2 launcher and enter the new Laptop 1 IPv4 address. If Windows presents a firewall prompt, allow Python on the trusted private network you are using for the demo.

## Showcase the demo

1. On Laptop 1, show the API terminal with the mode, model and ten-trip count. On Laptop 2, show the matching mode and live trip menu.
2. Let the evaluator choose any trip and click **Run live prediction**. Laptop 2 sends the trip ID to `POST /api/predict`.
3. On Laptop 2, show the teal predicted route, blue GPS points and grey roads. Toggle the layers if they overlap. Point out that the metrics are for the selected trip. The **0.9127 Macro F1** is the prior aggregate student-test result across ten trips.
4. On Laptop 1, find the matching **Request ID** in the terminal. Show the inference entry and the successful HTTP response. Repeat with a different trip to demonstrate live processing.
5. Capture the working UI and the successful API entry in the receiving terminal for the technical PDF. Include a separate student-data Folium view in the report. That report view may use the supplied student reference route. Keep evaluator reference labels off the UI.

## Guide references

- **Main Guidelines**, sections 3–4, defines the laptop split and FastAPI integration.
- **Models and Training Guidelines**, final evaluation section, specifies ten selectable reserved trips, live inference, server-side scoring, and receiving-laptop logs.
- **Dataset Exploration**, sections 3 and 7, describes the trip files and the partition boundaries.
- **Assessment and Plagiarism Policy**, data leakage section, prohibits evaluator labels in prediction or the UI and prohibits hardcoded predictions.
- **Deliverables**, sections 6–7, specifies the Folium report view and the screenshots of the UI and receiving terminal.
