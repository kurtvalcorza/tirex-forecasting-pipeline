import sys
import types

import numpy as np

from tirex_forecasting_pipeline import TiRexForecastPipeline, last_value_baseline, mae


class FakeTimeseriesType:
    def __init__(self, *, target, past_covariates, future_covariates):
        self.target = target
        self.past_covariates = past_covariates
        self.future_covariates = future_covariates


class FakeModel:
    def forecast(self, timeseries, prediction_length, output_type):
        assert output_type == "numpy"
        n_variates = timeseries[0].target.shape[0]
        return [np.zeros((n_variates, 9, prediction_length))]


def test_shapes(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "tirex2",
        types.SimpleNamespace(TimeseriesType=FakeTimeseriesType),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(from_numpy=lambda value: value),
    )
    pipeline = TiRexForecastPipeline(FakeModel(), "cpu")
    result = pipeline.forecast(np.arange(64), horizon=8)
    assert result["median"].shape == (1, 8)
    assert result["quantiles"].shape == (1, 9, 8)


def test_baseline():
    baseline = last_value_baseline([1, 2, 3], 2)
    assert baseline.tolist() == [[3, 3]]
    assert mae([3, 4], [3, 3]) == 0.5
