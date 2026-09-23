---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: time-series-forecasting
task: "Others - Time-Series Forecasting"
base_model: NX-AI/TiRex-2
date_published: "2026-06-16"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt`, https://huggingface.co/api/models/NX-AI/TiRex-2)"
---

# TiRex-2 — Time-Series Foundation Model (Zero-Shot Probabilistic Forecasting)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-NX--AI%2FTiRex--2-ffcc4d?style=flat)](https://huggingface.co/NX-AI/TiRex-2)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-NX--AI%2Ftirex--2-181717?style=flat&logo=github&logoColor=white)](https://github.com/NX-AI/tirex-2)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2607.01204-b31b1b.svg)](https://arxiv.org/abs/2607.01204)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://huggingface.co/NX-AI/TiRex-2/blob/05e5b26db52bfb256f1ae1bdf785589850482de3/LICENSE)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — bootstrap a fresh runtime, resolve and verify the pinned upstream revision, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/tirex-forecasting-pipeline/blob/main/tutorials/tirex_forecasting_colab.ipynb) [`tirex_forecasting_colab.ipynb`](https://github.com/kurtvalcorza/tirex-forecasting-pipeline/blob/main/tutorials/tirex_forecasting_colab.ipynb)  
  *Zero-shot probabilistic forecasting with the pinned `NX-AI/TiRex-2` checkpoint and chronological evaluation on bundled or your own series; no training occurs.*

---

#### Description

TiRex-2 is NX-AI's pretrained time-series foundation model for zero-shot univariate and multivariate forecasting, packaged here from `NX-AI/TiRex-2` at immutable revision `05e5b26db52bfb256f1ae1bdf785589850482de3`. The open model forecasts one or more target variates from history and can condition on past and future-known covariates without task-specific training. This repository adds immutable loading, finite-shape and covariate-alignment validation, normalized quantile/median outputs, chronological evaluation helpers, baselines, provenance, and a tutorial contract.

#### Intended Use and Limitations

###### Primary Intended Uses

The primary task is zero-shot numerical time-series forecasting from historical target values, with optional multivariate targets and optional past or future-known covariates supported by the upstream model. Intended application domains include demand, operations, sensor telemetry, environmental series, scheduling-related data, and other ordered numerical processes where future values are held out for validation. The pipeline is intended as a strong zero-shot baseline or inference service, not a fine-tuning framework.

###### Primary Intended Users

Primary users are forecasting practitioners, ML engineers, data scientists, quantitative analysts, researchers, and application developers who understand chronological evaluation, leakage, target-versus-covariate semantics, prediction horizons, quantile forecasts, and the need to compare foundation-model forecasts with simple baselines. Users are expected to validate on their own historical periods and to know which future covariates are genuinely available at prediction time rather than derived from future targets.

###### Out-of-scope use cases

1. **Capability boundary:** the open TiRex-2 release in this repository does not expose fine-tuning, streaming updates, classification, or regression adaptations advertised separately as TiRex-2 Pro capabilities.
2. **Input boundary:** the pipeline wrapper requires finite 1D/2D target arrays, at least 32 and at most 16,384 context steps, and a horizon of 1–4,096 steps. Optional `past_covariates` must contain exactly the target context length, and optional `future_covariates` must contain exactly `context_length + horizon` steps; misaligned covariates are rejected before model execution.
3. **Decision boundary:** forecasts are not approved as sole inputs to high-consequence autonomous decisions without representative backtesting and human/domain oversight.

#### Factors

###### Groups

This pipeline is not inherently demographic: its core input is numerical time-series data. However, human-impacting datasets may encode groups indirectly through geography, service access, customer segments, or operational allocation. The upstream pretraining mixture is not independently group-audited by this repository. Operators using forecasts to allocate resources or services must define relevant groups in their own data and evaluate error and coverage differences across those groups before deployment.

###### Instrumentation

Time-series inputs may be produced by sensors, meters, transaction systems, monitoring agents, APIs, ETL jobs, business databases, or manually maintained operational records. Sampling interval, clock alignment, aggregation, unit changes, sensor drift, backfills, outages, and changed ETL logic can all become forecasting error. The wrapper validates dimensionality, finite values, and covariate time-axis alignment but cannot determine whether upstream instruments were calibrated or whether a historical regime change invalidates the learned prior.

###### Environment

The portable reference environment is Python 3.12 with `tirex-2==0.2.1`, PyTorch 2.8, NumPy 2.3.3, and pandas 2.3.3. CPU is the default tutorial device. Upstream CUDA execution requires compatible recent NVIDIA hardware plus a matching CUDA toolkit because fused recurrent kernels may compile on first use; upstream `xlstm` also resolves CUDA include paths at import whenever a GPU is visible, so CPU inference on a GPU-equipped machine without the toolkit needs the GPU hidden (`CUDA_VISIBLE_DEVICES=""`) or a resolvable toolkit (`CUDA_HOME`/`CUDA_PATH`, `nvcc`, `/usr/local/cuda`, as PyTorch resolves it), which the wrapper otherwise reports as a typed error. The data environment assumes temporally ordered numerical histories whose future evaluation period is not leaked into context or covariates; future-known covariates must genuinely be available through the requested horizon.

#### Metrics

###### Performance Measures

The repository reports `mae` and `rmse` on chronological holdouts. MAE measures average absolute error in the target's units and is easy to interpret, while RMSE weights larger errors more strongly and exposes a different failure mode. The tutorial also reports the same measures for a last-value baseline. The public `evaluation_report` stage writes these measures, the q10–q90 `interval_coverage` and the baseline comparison to a machine-readable report whose verdict is `sample-sanity` on the withheld tutorial holdout and `not-measurable` when no truth is supplied. These are tutorial/backtest metrics for the demonstrated series; upstream leaderboard numbers are not presented as measurements reproduced by this pipeline.

###### Decision thresholds

Forecasting does not apply a categorical decision threshold. The normalized point forecast is the model's q=0.5 quantile, and the wrapper also exposes q=0.1 through q=0.9 without converting intervals into alert or action labels. Deployment owners must define any operational threshold themselves using domain costs and representative backtests; a model quantile is not a guaranteed frequentist confidence bound or a universal decision threshold.

###### Approaches to uncertainty and variability

The tutorial uses one chronological holdout and reports single MAE/RMSE values rather than cross-run dispersion. TiRex-2 returns nine model quantiles, which describe the model's predictive distribution but are not claimed as calibrated coverage guarantees in a new domain. Hardware kernels, package versions, and data windows can affect outputs. Operators needing calibrated uncertainty should measure empirical interval coverage over representative rolling or blocked backtests and recalibrate if necessary.

#### Ethical considerations and biases

###### Data

Upstream identifies training mixtures that include public time-series collections such as Chronos datasets and LOTSA-derived data, with details bounded by the upstream paper and model card. This repository does not independently enumerate or audit every upstream series for sensitivity. It distributes code, documentation, and tutorial logic but not user datasets. Operators must audit inference series and covariates for personal, confidential, proprietary, or restricted information and verify they are authorized to process them.

###### Human Life

The pipeline is not intended or certified for autonomous decisions in health, safety, criminal justice, employment, credit, housing, or other high-impact settings. No external board has validated this wrapper for such decisions. If forecasting supports a sensitive operational process, admissible use requires qualified human oversight, representative historical backtesting, monitoring for regime change, documented fallback procedures, and any domain-specific regulatory or institutional review that applies.

###### Mitigations

Implemented controls include an immutable Hugging Face revision passed through upstream `hf_kwargs`; a committed `dimer-base-manifest.json` whose per-file SHA-256 digests `verify_snapshot` re-checks before every load; the public `validate_inputs` stage, which applies the same target, horizon and covariate checks as `forecast` and writes an input manifest with any rejection recorded as a finding; exact `tirex-2` and core runtime pins; finite target and covariate checks; exact past/future covariate time-axis validation; explicit context and horizon ceilings; normalized quantile ordering; explicit q=0.5 median semantics; chronological tutorial backtesting; pre-pandas duplicate CSV-header rejection; last-value comparison; machine-readable repository/model provenance; upstream `weights_only=True` checkpoint deserialization; unit tests around output, metric, and covariate-shape contracts; and source validation that prevents notebook/model-card placeholders from shipping unnoticed.

###### Risks and harms

Forecasts can be inaccurate under distribution shift, regime changes, sparse history, bad covariates, sensor/ETL errors, or domains unlike upstream pretraining. Future-target leakage can make retrospective results falsely optimistic. Quantiles may be misread as calibrated guarantees. Forecast errors can cause inventory, staffing, capacity, or resource-allocation harms, particularly when automated. Covariates can also encode sensitive attributes or interventions incorrectly. These risks rise when simple baselines and representative chronological backtests are omitted.

###### Use cases

The pipeline must not be used for social scoring, unlawful discrimination, deceptive manipulation, covert surveillance, or automated denial/allocation of essential services based solely on an unvalidated forecast. It must not be used to smuggle future target information into evaluation covariates or to present leaked backtests as independent evidence. Use must comply with the Apache-2.0 upstream license, data-source terms, privacy obligations, and the policy of the deployment that runs the pipeline.

## Immutable provenance

- Model: `NX-AI/TiRex-2`
- Revision: `05e5b26db52bfb256f1ae1bdf785589850482de3`
- Runtime package: `tirex-2==0.2.1`
- Snapshot manifest: `weights/tirex-2/dimer-base-manifest.json` — `model.ckpt` SHA-256 `184b160ffbe4c01a26beeba14015ff3507c7497e1f3577114187bbc1d19fcac1` (380613375 bytes)
- Upstream repository: https://github.com/NX-AI/tirex-2
- Paper: https://arxiv.org/abs/2607.01204
