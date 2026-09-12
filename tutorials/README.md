# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/tirex-forecasting-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tirex-forecasting-pipeline/blob/main/tutorials/tirex_forecasting_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-NX--AI%2FTiRex--2-ffcc4d?style=flat)](https://huggingface.co/NX-AI/TiRex-2)
[![Upstream](https://img.shields.io/badge/Upstream-NX--AI%2Ftirex--2-181717?style=flat&logo=github&logoColor=white)](https://github.com/NX-AI/tirex-2)
[![arXiv](https://img.shields.io/badge/arXiv-2607.01204-b31b1b.svg)](https://arxiv.org/abs/2607.01204)
[![Model released](https://img.shields.io/badge/Model%20released-2026--06--16-6f42c1?style=flat)](https://huggingface.co/NX-AI/TiRex-2/tree/05e5b26db52bfb256f1ae1bdf785589850482de3)
[![Sample backtest](https://img.shields.io/badge/Sample%20backtest-MAE%200.0491%20%7C%20RMSE%200.0646-2ea44f?style=flat)](../docs/release-verification.md)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../LICENSE)

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `tirex_forecasting_colab.ipynb` | `TASK-INFERENCE` | Zero-shot univariate/multivariate probabilistic forecasting (covariate contract validated in unit tests) | CPU (`device='cpu'`) | CSV with `timestamp` + numeric targets, gated off by default | **Candidate** — static checks pass; clean-runtime execution evidence is recorded in `../docs/release-verification.md` and must be reviewed for the exact notebook revision before promotion |

## Conformance notes

- The notebook exercises `TiRexForecastPipeline` from the repository public API; the pipeline passes the immutable revision to the upstream resolver.
- CPU is the reference path because upstream CUDA execution compiles fused recurrent kernels on first use; the notebook does not assume a GPU.
- BYOD CSV headers are inspected before pandas ingestion so duplicate columns cannot be silently renamed; timestamps must be unique, increasing and regularly spaced.
- `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the
  clean-runtime execution requirement; a release review must confirm that a recorded clean run in
  `docs/release-verification.md` matches the notebook revision under review before the status is
  promoted to `Release-grade`.
