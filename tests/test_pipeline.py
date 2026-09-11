import sys
import types

import numpy as np
import pytest

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


def _stub_runtime(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "tirex2",
        types.SimpleNamespace(TimeseriesType=FakeTimeseriesType),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            from_numpy=lambda value: value,
            cuda=types.SimpleNamespace(is_available=lambda: False),
        ),
    )


def test_shapes(monkeypatch):
    _stub_runtime(monkeypatch)
    pipeline = TiRexForecastPipeline(FakeModel(), "cpu")
    result = pipeline.forecast(np.arange(64), horizon=8)
    assert result["median"].shape == (1, 8)
    assert result["quantiles"].shape == (1, 9, 8)


def test_covariate_shapes(monkeypatch):
    _stub_runtime(monkeypatch)
    pipeline = TiRexForecastPipeline(FakeModel(), "cpu")
    result = pipeline.forecast(
        np.arange(64),
        horizon=8,
        past_covariates=np.ones((2, 64)),
        future_covariates=np.ones((3, 72)),
    )
    assert result["past_covariates"] == 2
    assert result["future_covariates"] == 3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"past_covariates": np.ones((1, 63))}, "past_covariates must contain exactly 64"),
        ({"future_covariates": np.ones((1, 71))}, "future_covariates must contain exactly 72"),
    ],
)
def test_rejects_misaligned_covariates(monkeypatch, kwargs, match):
    _stub_runtime(monkeypatch)
    pipeline = TiRexForecastPipeline(FakeModel(), "cpu")
    with pytest.raises(ValueError, match=match):
        pipeline.forecast(np.arange(64), horizon=8, **kwargs)


def test_baseline():
    baseline = last_value_baseline([1, 2, 3], 2)
    assert baseline.tolist() == [[3, 3]]
    assert mae([3, 4], [3, 3]) == 0.5


def _stub_cuda_toolkit(monkeypatch, cuda_home):
    """Model torch's resolved toolkit path (torch.utils.cpp_extension.CUDA_HOME)."""
    sys.modules["torch"].cuda = types.SimpleNamespace(is_available=lambda: True)
    cpp_extension = types.SimpleNamespace(CUDA_HOME=cuda_home)
    monkeypatch.setitem(sys.modules, "torch.utils", types.SimpleNamespace(cpp_extension=cpp_extension))
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", cpp_extension)


@pytest.mark.parametrize("device", ["cpu", "cpu:0"])
def test_cpu_path_with_visible_gpu_and_no_toolkit_is_a_typed_error(monkeypatch, device):
    _stub_runtime(monkeypatch)
    _stub_cuda_toolkit(monkeypatch, cuda_home=None)
    with pytest.raises(RuntimeError, match="no CUDA toolkit was found"):
        TiRexForecastPipeline.from_pretrained(device=device)


def test_cpu_path_with_resolved_toolkit_reaches_the_loader(monkeypatch):
    _stub_runtime(monkeypatch)
    _stub_cuda_toolkit(monkeypatch, cuda_home="/usr/local/cuda")
    sys.modules["tirex2"].load_model = lambda *args, **kwargs: FakeModel()
    pipeline = TiRexForecastPipeline.from_pretrained(device="cpu")
    assert isinstance(pipeline, TiRexForecastPipeline)
