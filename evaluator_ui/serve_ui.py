"""Serve the evaluator UI and forward its three API calls from Laptop 2."""
import argparse
from http.client import HTTPConnection, HTTPException
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Timer
from urllib.parse import urlsplit
import webbrowser

WEB = Path(__file__).resolve().parent / "web"
GET_ENDPOINTS = {"/api/health", "/api/trips"}
MAX_RESPONSE = 16 * 1024 * 1024


def parse_backend(value):
    address = urlsplit(value.strip().rstrip("/"))
    if (address.scheme != "http" or not address.hostname or address.username or address.password
            or address.path or address.query or address.fragment):
        raise ValueError("Enter a plain HTTP address such as http://192.168.1.12:8000.")
    return address, address.port or 80


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def json_response(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def forward(self, body=None):
        host, port = self.server.backend
        connection = HTTPConnection(host, port, timeout=25)
        try:
            connection.request(self.command, self.path, body=body,
                               headers={"Content-Type": "application/json"} if body is not None else {})
            response = connection.getresponse()
            data = response.read(MAX_RESPONSE + 1)
            if len(data) > MAX_RESPONSE:
                raise ValueError("API response is too large")
            message = json.loads(data)
        except (OSError, HTTPException, ValueError):
            self.json_response(502, {"detail": "Laptop 1 did not answer. Check its API window, LAN address and network."})
            return
        finally:
            connection.close()
        self.json_response(response.status, message)

    def do_GET(self):
        if self.path in GET_ENDPOINTS:
            self.forward()
        elif self.path == "/client-config.json":
            self.json_response(200, {"backend_url": self.server.backend_url})
        elif self.path.startswith("/api/"):
            self.json_response(404, {"detail": "Unknown evaluator endpoint."})
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != "/api/predict":
            self.json_response(404, {"detail": "Unknown evaluator endpoint."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= 1024:
            self.json_response(400, {"detail": "Expected a trip selection."})
            return
        self.forward(self.rfile.read(length))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", help="Laptop 1 URL, for example http://192.168.1.12:8000.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the UI in the default browser.")
    args = parser.parse_args()
    backend = args.api or input("Laptop 1 API address (include http:// and the port): ")
    try:
        address, port = parse_backend(backend)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    server.backend = address.hostname, port
    server.backend_url = address.geturl()
    ui_url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Evaluator UI: {ui_url}\nLaptop 1: {server.backend_url}\nCtrl+C stops this UI.", flush=True)
    if args.open:
        Timer(.25, webbrowser.open, args=(ui_url,)).start()
    with server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
