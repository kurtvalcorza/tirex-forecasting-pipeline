# ruff: noqa: E501  -- single-line test fixtures and assertions are kept readable
"""The model is served from a pinned safetensors conversion, never from the pickle checkpoint.

The offline tests need no model library. The last test compares ``tirex2.load_model`` on the pinned
``model.ckpt`` with the safetensors loader on CPU; it runs only where the checkpoint is staged under
``weights/`` and ``tirex2``/``safetensors`` are installed (CI installs neither).
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from tirex_forecasting_pipeline import pipeline as P


def test_digests_are_pinned_and_the_source_matches_the_manifest() -> None:
    manifest = json.loads((P.DEFAULT_WEIGHTS_DIR / P.MANIFEST_NAME).read_text(encoding="utf-8"))
    entries = {e["path"]: e["sha256"] for e in manifest["files"]}
    assert entries[P.WEIGHTS_FILE] == P.SOURCE_SHA256
    assert len(P.CONVERTED_SHA256) == 64 and int(P.CONVERTED_SHA256, 16) >= 0
    assert P.CONVERTED_SHA256 != P.SOURCE_SHA256
    assert P.CONVERTED_FILE == "model.safetensors"


def test_canonical_safetensors_sorts_the_header_and_keeps_the_data() -> None:
    header = {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "__metadata__": {"z": "1", "a": "2"}}
    text = json.dumps(header).encode()
    data = struct.pack("<Q", len(text)) + text + b"\x00\x00\x80\x3f"
    out = P.canonical_safetensors(data)
    size = struct.unpack("<Q", out[:8])[0]
    assert size % 8 == 0
    assert out[8 : 8 + size].rstrip(b" ") == json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    assert out[8 + size :] == b"\x00\x00\x80\x3f"
    assert P.canonical_safetensors(out) == out


def test_convert_refuses_a_source_that_misses_its_digest(tmp_path: Path) -> None:
    source = tmp_path / "model.ckpt"
    source.write_bytes(b"not the pinned checkpoint")
    with pytest.raises(ValueError, match="refusing to deserialise"):
        P.convert_checkpoint(source, tmp_path / "model.safetensors")
    assert not (tmp_path / "model.safetensors").exists()


def test_ensure_converted_refuses_a_conversion_that_misses_the_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(P, "stage_missing_files", lambda *a, **k: [])
    monkeypatch.setattr(P, "verify_snapshot", lambda *a, **k: {})

    def fake_convert(source, destination, **kwargs):
        Path(destination).write_bytes(b"other bytes")
        return {"converted_sha256": hashlib.sha256(b"other bytes").hexdigest()}

    monkeypatch.setattr(P, "convert_checkpoint", fake_convert)
    with pytest.raises(ValueError, match="converted sha256"):
        P.ensure_converted(tmp_path)
    assert not (tmp_path / P.CONVERTED_FILE).exists()


def test_the_pinned_conversion_reproduces_the_ckpt_forecast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On the real pinned checkpoint, when staged: two conversions give the pinned bytes, and the forecast equals
    tirex2.load_model's (eager mode, so no C++ compiler is needed) with torch.load refused on the serving path."""
    root = P.DEFAULT_WEIGHTS_DIR
    if not (root / P.WEIGHTS_FILE).is_file():
        pytest.skip("the pinned checkpoint is not staged under weights/")
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    tirex2 = pytest.importorskip("tirex2")
    dynamo = pytest.importorskip("torch._dynamo")
    monkeypatch.setattr(dynamo.config, "disable", True)
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    assert P.convert_checkpoint(root / P.WEIGHTS_FILE, a)["converted_sha256"] == P.CONVERTED_SHA256
    P.convert_checkpoint(root / P.WEIGHTS_FILE, b)
    assert a.read_bytes() == b.read_bytes()
    stock = tirex2.load_model(str(root), device="cpu")
    real_load = torch.load
    monkeypatch.setattr(torch, "load", lambda *x, **k: pytest.fail("torch.load called on the serving path"))
    served = P.load_converted_model(root, device="cpu", weights=a)
    monkeypatch.setattr(torch, "load", real_load)
    t = np.arange(224, dtype=np.float32)
    series = (0.02 * t + np.sin(t / 8)).astype(np.float32)[None, :]
    ts = [tirex2.TimeseriesType(target=torch.from_numpy(series), past_covariates=None, future_covariates=None)]
    with torch.no_grad():
        expected = stock.forecast(ts, prediction_length=32, output_type="numpy")[0]
        got = served.forecast(ts, prediction_length=32, output_type="numpy")[0]
    assert np.array_equal(got, expected)


def test_the_validator_still_refuses_any_other_torch_load() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("validate_release_assets", Path(__file__).resolve().parents[1] / "tools" / "validate_release_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pattern = dict(module.FORBIDDEN_PATTERNS)["unsafe deserialization"]
    assert pattern.search("torch.load(path)")
    assert pattern.search("torch.load(path, weights_only=False)")
    assert pattern.search("pickle.load(fh)")
    assert not pattern.search('torch.load(source, map_location="cpu", weights_only=True)')
