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
result = pipe.forecast([1, 2, 3, 4] * 16, horizon=12)
print(result["median"])
```

For covariate-conditioned forecasting, the DIMER wrapper enforces the upstream time-axis contract before model execution: `past_covariates` must have exactly the target context length, while `future_covariates` must contain exactly `context_length + horizon` steps. Both accept shape `(n_covariates, time)` or a one-dimensional single-covariate input and must contain finite values.

## Tutorial

`tutorials/tirex_forecasting_colab.ipynb` is `TASK-INFERENCE`. It self-bootstraps in a fresh runtime, validates raw BYOD CSV headers before pandas ingestion, uses chronological backtesting, MAE/RMSE, a last-value baseline, explicit horizon/context semantics, probabilistic quantiles, and JSON/CSV provenance export.

## Release status

**Candidate.** Full clean-runtime execution evidence remains required; local pre-flight runs are recorded in `docs/release-verification.md`. CPU is the portable reference path; CUDA has additional upstream compiler/kernel requirements and is not the default tutorial assumption. On a machine where a CUDA GPU is visible but no CUDA toolkit is installed, upstream `xlstm` fails at import even for `device="cpu"`; `from_pretrained` raises a typed `RuntimeError` naming the fix (hide the GPU with `CUDA_VISIBLE_DEVICES=""` before importing torch, or set `CUDA_HOME`).
