# TiRex-2 Forecasting Pipeline

DIMER-oriented zero-shot forecasting wrapper for **NX-AI TiRex-2**, supporting univariate and multivariate targets plus optional past and future-known covariates through the upstream open inference API.

## Upstream alignment

- Model: `NX-AI/TiRex-2`
- Revision: `05e5b26db52bfb256f1ae1bdf785589850482de3`
- Runtime package: `tirex-2==0.2.1`
- Weight license: Apache-2.0
- Open-release capability: zero-shot forecasting; **no task-specific fine-tuning in this repository**
- Quantile output: 0.1 through 0.9 in 0.1 increments; median is q=0.5

## Public API

```python
from tirex_forecasting_pipeline import TiRexForecastPipeline
pipe = TiRexForecastPipeline.from_pretrained(device="cpu")
result = pipe.forecast([1,2,3,4] * 16, horizon=12)
print(result["median"])
```

## Tutorial

`tutorials/tirex_forecasting_colab.ipynb` is `TASK-INFERENCE`. It uses chronological backtesting, MAE/RMSE, a last-value baseline, explicit horizon/context semantics, probabilistic quantiles, optional BYOD CSV, and JSON/CSV export.

## Release status

**Candidate.** Full clean-runtime execution evidence remains required. CPU is the portable reference path; CUDA has additional upstream compiler/kernel requirements and is not the default tutorial assumption.
