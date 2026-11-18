"""Start the PathSnap FastAPI server on Laptop 1."""
import argparse
import logging
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from threadpoolctl import threadpool_limits
import uvicorn

from pathsnap.demo_data import TripSource
from pathsnap.demo_service import run_trip
from pathsnap.phase2 import load_decoder
from pathsnap.pipeline import MODEL_DIR, THREADS, load_models

MODEL_NAME = "Phase 2 LightGBM + Viterbi"
LOGGER = logging.getLogger("uvicorn.error")


class PredictionRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def build_app(source):
    directory = MODEL_DIR / "phase2-tuned"
    if not (directory / "decoder.json").is_file():
        raise ValueError("The frozen Phase 2 decoder is missing.")
    models, settings = load_models(directory), load_decoder(directory)
    app = FastAPI(title="PathSnap evaluator API", version="1.0")
    inference_lock = Lock()

    @app.get("/api/health")
    def health():
        return {"status": "ready", "mode": source.mode, "model": MODEL_NAME, "trip_count": len(source.trips)}

    @app.get("/api/trips")
    def trips():
        return {"mode": source.mode, "trip_ids": list(source.trips)}

    @app.post("/api/predict")
    def predict(request: PredictionRequest):
        if request.trip_id not in source.trips:
            raise HTTPException(status_code=404, detail="This trip is unavailable in the current dataset.")
        request_id = uuid4().hex[:10]
        start = perf_counter()
        LOGGER.info("PathSnap request=%s mode=%s trip=%s start", request_id, source.mode, request.trip_id)
        try:
            # Serialize CPU inference and configure the local BLAS thread budget.
            with inference_lock, threadpool_limits(limits=THREADS):
                response = run_trip(source, request.trip_id, models, settings)
        except (OSError, ValueError):
            LOGGER.exception("PathSnap request=%s trip=%s failed", request_id, request.trip_id)
            raise HTTPException(status_code=422, detail="This trip's server-side files could not be processed.")
        metrics = response["metrics"]
        LOGGER.info("PathSnap request=%s trip=%s complete observations=%s top1=%s macro_f1=%s elapsed_ms=%.1f",
                    request_id, request.trip_id, response["observations"],
                    f"{metrics['top1']:.4f}" if metrics else "unscored",
                    f"{metrics['macro_f1']:.4f}" if metrics else "unscored",
                    (perf_counter()-start)*1000)
        return {**response, "request_id": request_id, "model": MODEL_NAME}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Student ZIP or organiser-supplied ZIP/folder.")
    parser.add_argument("--mode", choices=("rehearsal", "evaluation"), default="rehearsal")
    parser.add_argument("--references", type=Path, help="Optional separate directory of server-side .route files.")
    parser.add_argument("--host", default="0.0.0.0", help="Use 127.0.0.1 for a loopback-only check.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        source = TripSource(args.mode, args.data, args.references)
        app = build_app(source)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Loaded {MODEL_NAME} | {source.mode} | {len(source.trips)} trips", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, access_log=True)


if __name__ == "__main__":
    main()
