"""Offline integration proof. Fakes replace ONLY model loading and sampling.

These tests do not demonstrate GPU inference or watermark effectiveness.
"""

import io
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import desynth
from webapp import create_app


def upload_image():
    image = Image.fromarray(np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return image, buffer


@pytest.mark.parametrize("advanced", [False, True])
def test_upload_through_real_restoration_preview_and_download(tmp_path, monkeypatch, advanced):
    app = create_app(model_dir=tmp_path, output_dir=tmp_path / "output")
    processor = app.extensions["processor"]
    client = app.test_client()
    source, buffer = upload_image()
    sampled = Image.new("RGB", source.size, (40, 90, 120))
    calls = []
    pipe, embeds = object(), object()
    monkeypatch.setattr(processor, "prerequisites", lambda: [])
    monkeypatch.setattr(processor, "_load_runtime", lambda: (pipe, embeds))

    def test_only_fake_sample(actual_pipe, image, actual_embeds, **settings):
        assert actual_pipe is pipe and actual_embeds is embeds
        assert image.tobytes() == source.tobytes()
        calls.append(settings)
        return sampled

    monkeypatch.setattr(desynth, "_sample", test_only_fake_sample)
    data = {"image": (buffer, "not-the-pixels.png")}
    if advanced:
        data.update(seed="42", denoise="0.5", steps="10", restore_mode="edge",
                    restore_sigma="1.5", unsharp="0.2")
    response = client.post("/process", data=data)
    assert response.status_code == 200
    result = response.get_json()
    seed = int(result["seed"])
    assert 0 <= seed < 2**63
    assert calls == [{"denoise": 0.5 if advanced else desynth.DEFAULT_DENOISE,
                      "steps": 10 if advanced else desynth.DEFAULT_STEPS, "seed": seed}]
    if advanced:
        assert seed == 42
    expected = desynth._restore(
        sampled, source, sigma=1.5 if advanced else desynth.DEFAULT_RESTORE_SIGMA,
        mode="edge" if advanced else "gaussian", unsharp_strength=0.2 if advanced else 0,
    )
    original = client.get(result["original"])
    preview = client.get(result["output"])
    download = client.get(result["download"])
    assert Image.open(io.BytesIO(original.data)).tobytes() == source.tobytes()
    assert Image.open(io.BytesIO(preview.data)).tobytes() == expected.tobytes()
    assert preview.data == download.data
    assert preview.mimetype == "image/png"
    assert 'attachment; filename=desynth.png' == download.headers["Content-Disposition"]
    assert sorted(path.name for path in (tmp_path / "output").glob("*/*")) == ["original.png", "output.png"]
    # Durable previews remain available after the service object is recreated.
    reopened = create_app(model_dir=tmp_path, output_dir=tmp_path / "output").test_client()
    assert reopened.get(result["download"]).data == download.data


def test_lazy_health_and_missing_prerequisites(tmp_path):
    client = create_app(model_dir=tmp_path, output_dir=tmp_path / "output").test_client()
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json["model_loaded"] is False
    assert any(desynth.GGUF_TRANSFORMER.name in message for message in health.json["prerequisites"])
    assert b"Setup needed before processing" in client.get("/").data
    _, buffer = upload_image()
    response = client.post("/process", data={"image": (buffer, "input.png")})
    assert response.status_code == 503
    assert "Missing model" in response.json["error"]
    assert not (tmp_path / "output").exists()
    # CLI help still works without importing/downloading inference dependencies.
    cli = subprocess.run([sys.executable, "desynth.py", "--help"], capture_output=True, check=True)
    assert b"--restore-mode" in cli.stdout


def test_processing_error_and_serialized_admission(tmp_path, monkeypatch):
    app = create_app(model_dir=tmp_path, output_dir=tmp_path / "output")
    processor = app.extensions["processor"]
    monkeypatch.setattr(processor, "prerequisites", lambda: [])
    client = app.test_client()
    processor._lock.acquire()
    try:
        _, buffer = upload_image()
        response = client.post("/process", data={"image": (buffer, "input.png")})
        assert response.status_code == 409
        assert "Another image" in response.json["error"]
    finally:
        processor._lock.release()

    def fail_loading():
        raise RuntimeError("test-only inference failure")

    monkeypatch.setattr(processor, "_load_runtime", fail_loading)
    _, buffer = upload_image()
    response = client.post("/process", data={"image": (buffer, "input.png")})
    assert response.status_code == 500
    assert "Processing failed" in response.json["error"]
    assert not processor._lock.locked()
    assert not (tmp_path / "output").exists()


class PageControls(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def test_page_controls_and_javascript_wiring(tmp_path):
    client = create_app(model_dir=tmp_path, output_dir=tmp_path).test_client()
    page = client.get("/")
    parser = PageControls()
    parser.feed(page.text)
    controls = {attrs["name"]: attrs for _, attrs in parser.tags if "name" in attrs}
    assert "multiple" not in controls["image"]
    assert controls["denoise"]["value"] == str(desynth.DEFAULT_DENOISE)
    assert controls["steps"]["value"] == str(desynth.DEFAULT_STEPS)
    assert controls["restore_sigma"]["value"] == str(desynth.DEFAULT_RESTORE_SIGMA)
    assert any(tag == "details" and "open" not in attrs for tag, attrs in parser.tags)
    assert any(tag == "form" and attrs["action"] == "/process" for tag, attrs in parser.tags)
    js = client.get("/static/app.js").text
    for binding in ("new FormData(form)", "fetch(form.action", "original.src = result.original",
                    "output.src = result.output", "download.href = result.download",
                    "controls.disabled = true", "controls.disabled = false",
                    "'working'", "'completed'", "'error'"):
        assert binding in js
    assert "does not verify watermark removal" in page.text


def test_ci_contract_uses_only_lightweight_checks():
    workflow = Path(".github/workflows/ui.yml").read_text()
    commands = [line.strip().removeprefix("- run: ") for line in workflow.splitlines()
                if line.strip().startswith("- run: ")]
    assert commands == ["pip install -r requirements-test.txt",
                        "python -m pytest -q tests", "docker compose config --quiet"]
    assert "pull_request:" in workflow
    assert "-r requirements-web.txt" in Path("requirements-test.txt").read_text()
    assert "torch" not in Path("requirements-web.txt").read_text()
