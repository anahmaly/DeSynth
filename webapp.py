"""Single-process UI; model loading occurs only on an explicit Process request."""

from __future__ import annotations

import importlib.util
import math
import os
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from flask import Flask, abort, jsonify, render_template, request, send_file, url_for
from PIL import Image, ImageOps, UnidentifiedImageError

import desynth


@dataclass(frozen=True)
class Settings:
    seed: int
    denoise: float = desynth.DEFAULT_DENOISE
    steps: int = desynth.DEFAULT_STEPS
    restore_mode: str = "gaussian"
    restore_sigma: float = desynth.DEFAULT_RESTORE_SIGMA
    unsharp: float = 0.0

    @classmethod
    def from_form(cls, form):
        try:
            settings = cls(
                seed=int(form["seed"]) if form.get("seed") else secrets.randbits(63),
                denoise=float(form.get("denoise", desynth.DEFAULT_DENOISE)),
                steps=int(form.get("steps", desynth.DEFAULT_STEPS)),
                restore_mode=form.get("restore_mode", "gaussian"),
                restore_sigma=float(form.get("restore_sigma", desynth.DEFAULT_RESTORE_SIGMA)),
                unsharp=float(form.get("unsharp", 0)),
            )
        except (ValueError, TypeError) as exc:
            raise ValueError("Use numbers for the advanced settings.") from exc

        if not (
            0 <= settings.seed < 2**63
            and 1 <= settings.steps <= 100
            and math.isfinite(settings.denoise)
            and 0 < settings.denoise <= 1
            and int(settings.steps * settings.denoise) >= 1
            and settings.restore_mode in ("gaussian", "edge")
            and 0 < settings.restore_sigma <= 10
            and 0 <= settings.unsharp <= 2
        ):
            raise ValueError("Check advanced settings. Steps × denoise must be at least 1.")
        return settings


class Processor:
    def __init__(self, model_dir: Path):
        self.transformer = model_dir / desynth.GGUF_TRANSFORMER.name
        self.lora = model_dir / desynth.LIGHTNING_LORA.name
        self._runtime = None
        self._lock = threading.Lock()

    def prerequisites(self) -> list[str]:
        problems = [
            f"Missing model: {path.name}. Place it in the mounted models directory."
            for path in (self.transformer, self.lora, desynth.EMBEDS_CACHE)
            if not path.is_file()
        ]
        if importlib.util.find_spec("torch") is None:
            problems.append("PyTorch is missing. Use the supplied Docker image.")
        elif not Path("/dev/nvidiactl").exists():
            problems.append("NVIDIA GPU is not exposed. Start with the supplied GPU Compose configuration.")
        return problems

    def _load_runtime(self):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; check the NVIDIA driver and container runtime.")
        pipe = desynth.build_pipeline(self.transformer, self.lora)
        embeds = torch.load(desynth.EMBEDS_CACHE, map_location="cpu", weights_only=True)
        return pipe, embeds

    def process(self, image: Image.Image, settings: Settings) -> Image.Image:
        # One worker process owns this lock and the cached pipeline. Reject rather
        # than queue another expensive GPU request; health remains responsive.
        if not self._lock.acquire(blocking=False):
            raise BlockingIOError("Another image is processing. Try again when it finishes.")
        try:
            if self._runtime is None:
                self._runtime = self._load_runtime()
            pipe, embeds = self._runtime
            clean = desynth.run_once(
                pipe, image, embeds, denoise=settings.denoise,
                steps=settings.steps, seed=settings.seed, passes=1,
            )
            return desynth._restore(
                clean, image, sigma=settings.restore_sigma,
                mode=settings.restore_mode, unsharp_strength=settings.unsharp,
            )
        finally:
            self._lock.release()


def create_app(*, model_dir: Path | None = None, output_dir: Path | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
    outputs = output_dir or Path(os.environ.get("DESYNTH_OUTPUT_DIR", "web-output"))
    outputs = outputs.resolve()
    processor = Processor(model_dir or Path(os.environ.get("DESYNTH_MODEL_DIR", ".")))
    app.extensions["processor"] = processor

    @app.get("/")
    def index():
        return render_template("index.html", defaults=Settings(seed=0), problems=processor.prerequisites())

    @app.get("/health")
    def health():
        problems = processor.prerequisites()
        return jsonify(status="ok", prerequisites=problems, model_loaded=processor._runtime is not None)

    @app.post("/process")
    def process():
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify(error="Choose one image first."), 400
        try:
            settings = Settings.from_form(request.form)
            with Image.open(upload.stream) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError):
            return jsonify(error="Check the image and advanced settings. Use a valid image and steps × denoise ≥ 1."), 400

        problems = processor.prerequisites()
        if problems:
            return jsonify(error="Processing unavailable. " + " ".join(problems)), 503
        try:
            final = processor.process(image, settings)
            job_dir = outputs / uuid4().hex
            job_dir.mkdir(parents=True)
            image.save(job_dir / "original.png")
            final.save(job_dir / "output.png")
        except BlockingIOError as exc:
            return jsonify(error=str(exc)), 409
        except Exception:
            app.logger.exception("Image processing failed")
            return jsonify(error="Processing failed. Check GPU memory, model files and first-run Hugging Face access; see service logs for details."), 500

        return jsonify(
            original=url_for("artifact", job_id=job_dir.name, kind="original"),
            output=url_for("artifact", job_id=job_dir.name, kind="output"),
            download=url_for("artifact", job_id=job_dir.name, kind="output", download=1),
            seed=str(settings.seed),
        )

    @app.get("/images/<job_id>/<kind>")
    def artifact(job_id, kind):
        if len(job_id) != 32 or any(char not in "0123456789abcdef" for char in job_id):
            abort(404)
        if kind not in ("original", "output"):
            abort(404)
        path = outputs / job_id / f"{kind}.png"
        if not path.is_file():
            abort(404)
        return send_file(path, mimetype="image/png", as_attachment="download" in request.args,
                         download_name="desynth.png" if kind == "output" else "original.png")

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error="Choose an image smaller than 32 MB."), 413

    return app
