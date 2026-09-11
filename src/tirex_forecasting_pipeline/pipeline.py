from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .validation import validate_horizon, validate_target

MODEL_ID = "NX-AI/TiRex-2"
MODEL_REVISION = "05e5b26db52bfb256f1ae1bdf785589850482de3"
MODEL_LICENSE = "Apache-2.0"
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


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


@dataclass
class TiRexForecastPipeline:
    _model: Any
    device: str

    @classmethod
    def from_pretrained(cls, device: str = "cpu") -> TiRexForecastPipeline:
        import os

        import torch

        if device == "cpu" and torch.cuda.is_available() and not os.environ.get("CUDA_HOME"):
            # Upstream xlstm resolves CUDA include paths at import time whenever a GPU is
            # visible, even though CPU inference never compiles a kernel. Surface that as a
            # typed error instead of letting an OSError escape from deep inside the import.
            raise RuntimeError(
                "A CUDA device is visible but CUDA_HOME is unset; upstream xlstm needs the CUDA "
                "toolkit at import even for device='cpu'. Hide the GPU with "
                "CUDA_VISIBLE_DEVICES='' before importing torch, or set CUDA_HOME."
            )

        from tirex2 import load_model

        model = load_model(
            MODEL_ID,
            device=device,
            hf_kwargs={"revision": MODEL_REVISION},
        )
        return cls(model, device)

    def forecast(
        self,
        target,
        *,
        horizon: int,
        past_covariates=None,
        future_covariates=None,
    ) -> dict[str, Any]:
        values = validate_target(target)
        validate_horizon(horizon)
        context_length = values.shape[1]
        past_values = _validate_covariates(
            past_covariates,
            expected_length=context_length,
            name="past_covariates",
        )
        future_values = _validate_covariates(
            future_covariates,
            expected_length=context_length + horizon,
            name="future_covariates",
        )

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
            "median": quantiles[:, 4, :],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "horizon": horizon,
            "context_length": context_length,
            "n_variates": values.shape[0],
            "past_covariates": past_values.shape[0] if past_values is not None else 0,
            "future_covariates": future_values.shape[0] if future_values is not None else 0,
        }
