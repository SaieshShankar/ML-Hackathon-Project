# PathSnap evaluator UI for Laptop 2

This package contains the UI and bundled map assets. Laptop 1 runs the model and supplies the trip list, route and scores live. You only need Python 3 and a browser on Laptop 2.

1. Extract this ZIP.
2. On Windows, double-click START_UI.bat. When prompted, enter the HTTP address your teammate gives you, such as http://192.168.1.12:8000.
3. The browser opens at http://127.0.0.1:8765/. Wait for **Laptop 1 connected**.
4. Select any available trip and click **Run live prediction**. Use the map layer toggles to compare the GPS points and predicted route. The request ID also appears in Laptop 1's terminal.

Terminal alternative:

    python serve_ui.py --api http://LAPTOP1-IP:8000 --open

Leave the launcher open while using the UI. Restart it with the new address if Laptop 1's IP changes. The two laptops must be able to reach each other over the same network or hotspot.
