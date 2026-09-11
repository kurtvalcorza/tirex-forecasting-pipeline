from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .validation import validate_horizon, validate_target

MODEL_ID = "NX-AI/TiRex-2"
MODEL_REVISION = "05e5b26db52bfb256f1ae1bdf785589850482de3"
MODEL_LICENSE = "Apache-2.0"
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass
class TiRexForecastPipeline:
    _model: Any
    device: str

    @classmethod
    def from_pretrained(cls, device: str = "cpu") -> TiRexForecastPipeline:
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

        import torch
        from tirex2 import TimeseriesType

        def as_covariates(value):
            if value is None:
                return None
            array = np.asarray(value, dtype=np.float32)
            if array.ndim == 1:
                array = array[None, :]
            if array.ndim != 2 or not np.isfinite(array).all():
                raise ValueError("covariates must be finite 1D/2D arrays")
            return torch.from_numpy(array)

        timeseries = TimeseriesType(
            target=torch.from_numpy(values),
            past_covariates=as_covariates(past_covariates),
            future_covariates=as_covariates(future_covariates),
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
            "context_length": values.shape[1],
            "n_variates": values.shape[0],
        }
