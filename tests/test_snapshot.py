"""Offline tests for the fleet snapshot scheme: manifest-driven verification, staging, loading."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tirex_forecasting_pipeline import (
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    TiRexForecastPipeline,
    stage_missing_files,
    verify_snapshot,
)
from tirex_forecasting_pipeline.pipeline import MANIFEST_NAME

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_MANIFEST = ROOT / "weights" / MODEL_KEY / MANIFEST_NAME

FILES = {"README.md": b"# TiRex-2\n", "model-config.yaml": b"num_blocks: 12\n", "model.ckpt": b"\x00" * 64}


def _write_snapshot(root: Path, *, files: dict[str, bytes] | None = None, model_id: str = MODEL_ID) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    files = FILES if files is None else files
    entries = []
    for name, payload in files.items():
        entries.append({"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": MODEL_KEY,
        "modelId": model_id,
        "revision": MODEL_REVISION,
        "files": entries,
        "totalBytes": sum(e["bytes"] for e in entries),
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _materialise(root: Path, files: dict[str, bytes] | None = None) -> None:
    for name, payload in (FILES if files is None else files).items():
        (root / name).write_bytes(payload)


def _stub_torch(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    )


def test_committed_manifest_names_the_pinned_identity_and_loader_files() -> None:
    manifest = json.loads(COMMITTED_MANIFEST.read_text(encoding="utf-8"))
    identity = (manifest["modelId"], manifest["revision"], manifest["modelKey"])
    assert identity == (MODEL_ID, MODEL_REVISION, MODEL_KEY)
    paths = {entry["path"] for entry in manifest["files"]}
    assert {"model-config.yaml", "model.ckpt"} <= paths, "tirex2.load_model needs both files"
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
    assert manifest["totalBytes"] == sum(entry["bytes"] for entry in manifest["files"])


def test_verify_snapshot_accepts_matching_files(tmp_path: Path) -> None:
    manifest = _write_snapshot(tmp_path)
    _materialise(tmp_path)
    result = verify_snapshot(tmp_path)
    assert result["path"] == str(tmp_path)
    assert result["files"] == manifest["files"]


def test_verify_snapshot_rejects_tampered_digest(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    _materialise(tmp_path)
    (tmp_path / "model.ckpt").write_bytes(b"\x01" * 64)  # same size, different bytes
    with pytest.raises(ValueError, match="model.ckpt: sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_identity(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, model_id="someone/else")
    _materialise(tmp_path)
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(tmp_path)
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_stage_missing_files_fetches_only_absent_entries(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    _materialise(tmp_path, {"README.md": FILES["README.md"], "model-config.yaml": FILES["model-config.yaml"]})
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched: list[str] = []

    def downloader(relative_path: str, root: Path) -> None:
        fetched.append(relative_path)
        (root / relative_path).write_bytes(FILES[relative_path])

    assert stage_missing_files(tmp_path, allow_download=True, downloader=downloader) == ["model.ckpt"]
    assert fetched == ["model.ckpt"]
    assert stage_missing_files(tmp_path, allow_download=True, downloader=downloader) == []
    verify_snapshot(tmp_path)


def test_from_pretrained_loads_the_verified_directory(monkeypatch, tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    _materialise(tmp_path)
    calls = []

    class FakeModel:
        def forecast(self, timeseries, prediction_length, output_type):
            return [np.zeros((1, 9, prediction_length))]

    def load_model(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeModel()

    monkeypatch.setitem(sys.modules, "tirex2", types.SimpleNamespace(load_model=load_model))
    _stub_torch(monkeypatch)
    pipe = TiRexForecastPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)
    assert pipe.source == "local-snapshot"
    assert calls == [((str(tmp_path),), {"device": "cpu"})]


def test_from_pretrained_refuses_a_tampered_snapshot(monkeypatch, tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    _materialise(tmp_path)
    (tmp_path / "model-config.yaml").write_bytes(b"num_blocks: 13\n")
    monkeypatch.setitem(sys.modules, "tirex2", types.SimpleNamespace(load_model=lambda *a, **k: object()))
    _stub_torch(monkeypatch)
    with pytest.raises(ValueError, match="model-config.yaml"):
        TiRexForecastPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)
