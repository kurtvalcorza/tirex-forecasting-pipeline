# Tutorials

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
