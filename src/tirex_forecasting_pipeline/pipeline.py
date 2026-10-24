"""Zero-shot probabilistic forecasting with the pinned ``NX-AI/TiRex-2`` checkpoint.

The model is served only from a digest-verified local snapshot (``weights/<key>/``:
``model-config.yaml`` + ``model.ckpt``, staged at the pinned revision). The pickle checkpoint is read
once, after its digest check, with ``torch.load(weights_only=True)`` and converted to
``model.safetensors`` (byte-reproducible, SHA-256 pinned in ``CONVERTED_SHA256``). The network is
then built from ``model-config.yaml`` exactly as ``tirex2.load_model`` builds it and loaded from the
safetensors file with ``load_state_dict(strict=True)``; nothing deserialises the pickle again and this
module never executes model-repository code.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import interval_coverage, last_value_baseline, mae, rmse
from .validation import MAX_CONTEXT, MAX_HORIZON, MIN_CONTEXT, validate_horizon, validate_target

MODEL_ID = "NX-AI/TiRex-2"
MODEL_REVISION = "05e5b26db52bfb256f1ae1bdf785589850482de3"
MODEL_LICENSE = "Apache-2.0"
MODEL_KEY = "tirex-2"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
CONFIG_FILE = "model-config.yaml"
WEIGHTS_FILE = "model.ckpt"
# SHA-256 of model.ckpt, equal to its manifest entry
SOURCE_SHA256 = "184b160ffbe4c01a26beeba14015ff3507c7497e1f3577114187bbc1d19fcac1"
CONVERTED_FILE = "model.safetensors"
CONVERTED_SHA256 = "7a97f73883d8466552afa79347d44f7730ca91dd82aa345606e8d339dbbe0c72"
CONVERSION_FORMAT = "tirex2-state-dict-safetensors-v1"
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MEDIAN_INDEX = 4  # QUANTILES[4] == 0.5: the point forecast is the model median, not a mean
POINT_FORECAST = "median (q=0.5)"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the checkpoint). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def canonical_safetensors(data: bytes) -> bytes:
    """Re-emit a safetensors header as sorted, compact JSON padded to 8 bytes; the data is unchanged.

    ``safetensors.torch.save`` (0.8.0) writes ``__metadata__`` in hash-map order, so equal inputs can give
    different bytes. Tensor offsets are relative to the end of the header, so re-ordering keeps them valid."""
    size = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + size])
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 8)
    return len(text).to_bytes(8, "little") + text + data[8 + size :]


def convert_checkpoint(
    source: str | Path, destination: str | Path, *, expected_sha256: str = SOURCE_SHA256
) -> dict[str, Any]:
    """Convert ``model.ckpt`` to safetensors once; return the source and converted digests.

    The source must match ``expected_sha256`` before it is read with
    ``torch.load(map_location="cpu", weights_only=True)``. A ``state_dict`` wrapper is unwrapped the way
    ``tirex2.load_model`` does; tensors are written as contiguous CPU clones with a canonical header."""
    source, destination = Path(source), Path(destination)
    source_sha = _sha256(source)
    if source_sha != expected_sha256:
        raise ValueError(
            f"{source.name}: sha256 {source_sha} != pinned {expected_sha256}; refusing to deserialise it"
        )
    import torch
    from safetensors.torch import save

    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    tensors = {key: value.detach().to("cpu").contiguous().clone() for key, value in state_dict.items()}
    metadata = {"format": CONVERSION_FORMAT, "source_file": source.name, "source_sha256": source_sha}
    data = canonical_safetensors(save(tensors, metadata=metadata))
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(destination)
    return {
        "source_sha256": source_sha,
        "converted_sha256": hashlib.sha256(data).hexdigest(),
        "tensors": len(tensors),
        "bytes": len(data),
    }


def ensure_converted(path: str | Path | None = None, *, allow_download: bool = False) -> Path:
    """Return the verified ``model.safetensors`` of a snapshot, converting the pinned source if needed.

    The snapshot (config included) is always staged and verified against its manifest. A present
    converted file whose SHA-256 equals ``CONVERTED_SHA256`` is used as is; otherwise ``model.ckpt``
    is converted once and the result must equal the pin."""
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    stage_missing_files(root, allow_download=allow_download)
    verify_snapshot(root)
    target = root / CONVERTED_FILE
    if target.is_file() and _sha256(target) == CONVERTED_SHA256:
        return target
    facts = convert_checkpoint(root / WEIGHTS_FILE, target)
    if facts["converted_sha256"] != CONVERTED_SHA256:
        target.unlink(missing_ok=True)
        raise ValueError(
            f"{CONVERTED_FILE}: converted sha256 {facts['converted_sha256']} != pinned {CONVERTED_SHA256}"
        )
    return target


def load_converted_model(root: str | Path, *, device: str, weights: str | Path | None = None) -> Any:
    """Build the network from ``model-config.yaml`` as ``tirex2.load_model`` (0.2.1) does and load the
    safetensors weights with ``strict=True``; return it wrapped in ``tirex2``'s ``ForecastModel``."""
    import torch
    import yaml
    from safetensors.torch import load_file
    from tirex2.api_adapter import ForecastModel
    from tirex2.model import TiRex2

    root = Path(root)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Execution on CUDA was requested but is not available.")
    with (root / CONFIG_FILE).open(encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)
    config["device"] = device
    model = TiRex2(**config)
    model.load_state_dict(load_file(str(weights or root / CONVERTED_FILE), device="cpu"), strict=True)
    return ForecastModel(model.eval())


def _validate_covariates(value, *, expected_length: int, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D with shape (covariates, time)")
    if array.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one covariate")
    if array.shape[1] != expected_length:
        raise ValueError(
            f"{name} must contain exactly {expected_length} time steps; got {array.shape[1]}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


INPUT_SCHEMA: dict[str, Any] = {
    "input": "1D array (time) or 2D array (variates, time) of finite numbers, cast to float32",
    "context_length": [MIN_CONTEXT, MAX_CONTEXT],
    "horizon": [1, MAX_HORIZON],
    "past_covariates": "optional (covariates, context_length), finite",
    "future_covariates": "optional (covariates, context_length + horizon), finite",
    "preprocessing": "none in this module; upstream tirex2 scales and patch-tokenises the context internally",
}


def _check_inputs(
    target: Any, horizon: int, past_covariates: Any = None, future_covariates: Any = None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Raise ValueError naming the first violated ceiling; return the validated arrays."""
    values = validate_target(target)
    validate_horizon(horizon)
    context_length = values.shape[1]
    past_values = _validate_covariates(
        past_covariates, expected_length=context_length, name="past_covariates"
    )
    future_values = _validate_covariates(
        future_covariates, expected_length=context_length + horizon, name="future_covariates"
    )
    return values, past_values, future_values


def validate_inputs(
    target: Any,
    *,
    horizon: int,
    past_covariates: Any = None,
    future_covariates: Any = None,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-variate observations, verdict).

    Rejection is reported by raising exactly as ``forecast`` would; a caller that wants the
    finding recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    values, past_values, future_values = _check_inputs(
        target, horizon, past_covariates, future_covariates
    )
    if names is not None and len(names) != values.shape[0]:
        raise ValueError("names must have one entry per variate")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {
                "id": names[i] if names else f"variate-{i}",
                "context_length": int(values.shape[1]),
                "min": float(values[i].min()),
                "max": float(values[i].max()),
            }
            for i in range(values.shape[0])
        ],
        "horizon": horizon,
        "past_covariates": int(past_values.shape[0]) if past_values is not None else 0,
        "future_covariates": int(future_values.shape[0]) if future_values is not None else 0,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any],
    truth: Any = None,
    *,
    context: Any = None,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``truth`` (the withheld future, shape ``(variates, horizon)``) the report carries the
    repository's ``mae`` / ``rmse`` on the median forecast and, when ``context`` is supplied, the
    same metrics for ``last_value_baseline`` plus ``interval_coverage`` of the q10–q90 band, with
    verdict ``sample-sanity``. Without ``truth`` the verdict is ``not-measurable``.
    """
    median = np.asarray(result["median"], dtype=float)
    base = {
        "task": "zero-shot probabilistic time-series forecasting",
        "point_forecast": POINT_FORECAST,
        "score_semantics": (
            "quantile levels 0.1..0.9 are model quantiles, not calibrated confidence intervals"
        ),
        "sample_kind": sample_kind,
        "horizon": int(result["horizon"]),
        "context_length": int(result["context_length"]),
        "n_variates": int(median.shape[0]),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if truth is None:
        return {
            **base,
            "metrics": [],
            "baselines": [],
            "verdict": "not-measurable",
            "reason": "no withheld future values were supplied for the forecast horizon",
            "needs": (
                "a chronological holdout: withhold the final `horizon` observations of the target, "
                "forecast from the remaining context, and score the median with mae/rmse against them "
                "and against the "
                "last_value_baseline, repeated over representative periods"
            ),
        }
    actual = np.asarray(truth, dtype=float)
    if actual.ndim == 1:
        actual = actual[None, :]
    if actual.shape != median.shape:
        raise ValueError(f"truth shape {actual.shape} != median shape {median.shape}")
    estimation = "single chronological holdout, no dispersion estimate"
    metrics = [
        {"id": "mae", "value": mae(actual, median), "estimation": estimation},
        {"id": "rmse", "value": rmse(actual, median), "estimation": estimation},
    ]
    baselines = []
    quantiles = np.asarray(result["quantiles"], dtype=float)
    levels = list(result["quantile_levels"])
    if quantiles.shape[:1] == actual.shape[:1] and 0.1 in levels and 0.9 in levels:
        metrics.append(
            {
                "id": "interval_coverage",
                "band": [0.1, 0.9],
                "value": interval_coverage(
                    actual, quantiles[:, levels.index(0.1), :], quantiles[:, levels.index(0.9), :]
                ),
                "nominal": 0.8,
                "estimation": estimation,
            }
        )
    if context is not None:
        baseline = last_value_baseline(context, int(result["horizon"]))
        baselines.append(
            {
                "id": "last_value_baseline",
                "metrics": [
                    {"id": "mae", "value": mae(actual, baseline)},
                    {"id": "rmse", "value": rmse(actual, baseline)},
                ],
            }
        )
    return {
        **base,
        "metrics": metrics,
        "baselines": baselines,
        "verdict": "sample-sanity",
        "reason": (
            f"one chronological holdout of {actual.shape[1]} step(s) on the tutorial sample; not a benchmark"
        ),
        "needs": (
            "repeated holdouts over representative periods of the deployment series for any "
            "generalisable claim"
        ),
    }


@dataclass
class TiRexForecastPipeline:
    _model: Any
    device: str
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str = "cpu",
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> TiRexForecastPipeline:
        import torch

        if device.startswith("cpu") and torch.cuda.is_available():
            # Upstream xlstm resolves CUDA include paths at import time whenever a GPU is
            # visible, even though CPU inference compiles no CUDA kernel (tirex-2's torch.compile of the
            # residual block does compile CPU code with TorchInductor, which needs a C++ compiler).
            # Use torch's own resolution (CUDA_HOME, CUDA_PATH, nvcc on PATH, /usr/local/cuda)
            # as the oracle and surface a typed error instead of letting an OSError escape from inside
            # the import.
            from torch.utils.cpp_extension import CUDA_HOME

            if CUDA_HOME is None:
                raise RuntimeError(
                    "A CUDA device is visible but no CUDA toolkit was found (CUDA_HOME/CUDA_PATH "
                    "unset, no nvcc on PATH, no /usr/local/cuda); upstream xlstm needs the toolkit "
                    "at import even for device='cpu'. Hide the GPU with CUDA_VISIBLE_DEVICES='' "
                    "before importing torch, or set CUDA_HOME."
                )

        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        if not (root / MANIFEST_NAME).is_file():
            # No Hub-direct load: the model is served only from the conversion of a verified snapshot.
            raise FileNotFoundError(
                f"no snapshot manifest at {root}; the model is served only from a verified snapshot. "
                f"Stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root} "
                f"and copy weights/{MODEL_KEY}/{MANIFEST_NAME} beside it"
            )
        weights = ensure_converted(root, allow_download=allow_download)
        model = load_converted_model(root, device=device, weights=weights)
        return cls(model, device, "local-snapshot")

    def forecast(
        self,
        target,
        *,
        horizon: int,
        past_covariates=None,
        future_covariates=None,
    ) -> dict[str, Any]:
        values, past_values, future_values = _check_inputs(
            target, horizon, past_covariates, future_covariates
        )
        context_length = values.shape[1]

        import torch
        from tirex2 import TimeseriesType

        timeseries = TimeseriesType(
            target=torch.from_numpy(values),
            past_covariates=(
                torch.from_numpy(past_values) if past_values is not None else None
            ),
            future_covariates=(
                torch.from_numpy(future_values) if future_values is not None else None
            ),
        )
        quantiles = np.asarray(
            self._model.forecast(
                [timeseries],
                prediction_length=horizon,
                output_type="numpy",
            )[0],
            dtype=float,
        )
        expected_shape = (values.shape[0], len(QUANTILES), horizon)
        if quantiles.shape != expected_shape:
            raise RuntimeError(
                f"unexpected TiRex forecast shape: {quantiles.shape}; "
                f"expected {expected_shape}"
            )
        return {
            "quantiles": quantiles,
            "quantile_levels": QUANTILES,
            "median": quantiles[:, MEDIAN_INDEX, :],
            "point_forecast": POINT_FORECAST,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "horizon": horizon,
            "context_length": context_length,
            "n_variates": values.shape[0],
            "past_covariates": past_values.shape[0] if past_values is not None else 0,
            "future_covariates": future_values.shape[0] if future_values is not None else 0,
            "device": self.device,
            "source": self.source,
        }
