"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package
modules, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "tirex_forecasting_pipeline",
    "repo_name": "tirex-forecasting-pipeline",
    "stem": "tirex_forecasting",
    "notebook_name": "tirex_forecasting_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "TiRexForecastPipeline",
    "weights_key": "tirex-2",
    "modules": ["pipeline.py", "validation.py", "evaluation.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "numpy", "pandas"],
    "title": "TiRex-2 — DIMER zero-shot forecasting tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/tirex-forecasting-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/tirex-forecasting-pipeline/blob/main/tutorials/tirex_forecasting_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-NX--AI%2FTiRex--2-ffcc4d?style=flat",
            "https://huggingface.co/NX-AI/TiRex-2",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-NX--AI%2Ftirex--2-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/NX-AI/tirex-2",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2607.01204-b31b1b.svg", "https://arxiv.org/abs/2607.01204"),
    ],
    "capability": "zero-shot probabilistic time-series forecasting with chronological evaluation, using the pinned `NX-AI/TiRex-2` checkpoint",
    "intro": (
        "At inference TiRex-2 maps a context window of one or more variates (plus optional past and future-known "
        "covariates) to nine forecast quantiles (q=0.1 … q=0.9) per step of the requested horizon in a single "
        "forward pass; the pipeline reports q=0.5 as the point forecast (a model median, not a mean) and keeps all "
        "nine quantiles. **No adaptation occurs:** no gradient training, fine-tuning, streaming adaptation or "
        "preprocessing fitting happens in this notebook — the upstream checkpoint supplies the weights and the "
        "model configuration, and the carried package adds snapshot verification, the input contract "
        "(`validate_target`, `validate_horizon`, the covariate time-axis rules), the `mae` / `rmse` / "
        "`last_value_baseline` / `interval_coverage` helpers and the public `validate_inputs` / `evaluation_report` "
        "stages. The default sample is a deterministic synthetic series generated in code and scored on one "
        "chronological holdout; its metrics are demonstration (plumbing) evidence, not a benchmark claim."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried package guarantees, resolve and digest-verify the "
        "immutable upstream checkpoint, generate a deterministic synthetic series (or upload a CSV), make a "
        "leakage-safe chronological holdout, validate the context into an input manifest, run the zero-shot "
        "forecast, read the median and the model quantiles correctly, produce an evaluation report that compares "
        "MAE/RMSE with a naive last-value baseline and reports q10–q90 coverage, and export machine-readable "
        "forecasts plus provenance."
    ),
    "exclusions": (
        "gradient training or fine-tuning, streaming/online adaptation, classification or regression on tabular "
        "features, anomaly detection, imputation, or calibrated prediction intervals. The quantiles are model "
        "quantiles, not guaranteed coverage; a single holdout is not a deployment-variability estimate."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The reference path is **CPU** (`device='cpu'`): upstream CUDA execution compiles fused recurrent kernels on first use and needs a CUDA toolkit, so the notebook does not assume a GPU. The pinned `torch==2.8.0` install is the largest download of the run.",
        "- **Knowledge:** basic Python and NumPy; what a forecast horizon, a context window and a quantile forecast are.",
        "- **Data:** the default sample is a deterministic synthetic trend + seasonal series (256 steps, fixed seed) generated in code, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one UTF-8 CSV with a unique `timestamp` column and one or more finite numeric target columns in chronological order, at least 64 rows (32 context + 32 holdout). Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Generate the synthetic sample or optional BYOD\n\n"
                "The default sample is **synthetic**: 256 steps of a linear trend plus a sine season plus small Gaussian "
                "noise from a fixed seed (`np.random.default_rng(7)`), so it needs no download and its SHA-256 is printed "
                "for the record. It has a real future — the final `HORIZON` steps are withheld in the next section — so "
                "the evaluation report can score the forecast, but a synthetic series says nothing about any deployment "
                "domain. BYOD is optional and disabled by default; when enabled, the raw CSV header is inspected before "
                "pandas reads the file so duplicate column names cannot be silently renamed, timestamps must parse, "
                "increase strictly and be regularly spaced, and the series must keep at least `MIN_CONTEXT` observations "
                "after the holdout is withheld. Look for a dictionary naming the sample kind, its shape and digest."
            ),
            "code": (
                "import csv\n"
                "import hashlib\n\n"
                "import pandas as pd\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "HORIZON = 32\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    sample_name = next(iter(uploaded))\n"
                "    with open(sample_name, newline='', encoding='utf-8-sig') as handle:\n"
                "        header = next(csv.reader(handle), [])\n"
                "    if not header or len(header) != len(set(header)):\n"
                "        raise ValueError('CSV must have non-empty unique column names; duplicate headers are rejected before pandas ingestion')\n"
                "    if 'timestamp' not in header:\n"
                "        raise ValueError('BYOD CSV must contain a timestamp column')\n"
                "    variate_names = [column for column in header if column != 'timestamp']\n"
                "    if not variate_names:\n"
                "        raise ValueError('BYOD CSV must contain at least one numeric target column')\n"
                "    frame = pd.read_csv(sample_name)\n"
                "    timestamps = pd.to_datetime(frame['timestamp'], errors='raise')\n"
                "    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():\n"
                "        raise ValueError('timestamps must be unique and strictly increasing')\n"
                "    if len(timestamps) > 2 and timestamps.diff().dropna().nunique() != 1:\n"
                "        raise ValueError('timestamps must be regularly spaced for this tutorial path')\n"
                "    series = frame[variate_names].to_numpy(dtype=float).T\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    rng = np.random.default_rng(7)\n"
                "    t = np.arange(256)\n"
                "    series = (0.02 * t + np.sin(t / 8) + rng.normal(0, 0.05, len(t)))[None, :]\n"
                "    variate_names = ['synthetic-trend-season']\n"
                "    sample_name = 'synthetic_trend_season_256.csv'\n"
                "    sample_kind = 'synthetic'\n\n"
                "series = validate_target(series)\n"
                "if series.shape[1] < HORIZON + MIN_CONTEXT:\n"
                "    raise ValueError(f'this tutorial needs at least {{HORIZON + MIN_CONTEXT}} rows ({{MIN_CONTEXT}} context + {{HORIZON}} holdout); got {{series.shape[1]}}')\n"
                "sample_sha256 = hashlib.sha256(np.ascontiguousarray(series, dtype=np.float32).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': sample_name, 'shape': list(series.shape), 'horizon': HORIZON, 'float32_sha256': sample_sha256}})"
            ),
        },
        {
            "md": (
                "## 5. Chronological holdout, then validate → input manifest\n\n"
                "The final `HORIZON` steps are **withheld** as the truth; only the earlier context is passed to the "
                "model, so no future target value leaks into the forecast. `validate_inputs` is the pipeline's public "
                "validation stage: it applies exactly the checks `forecast` applies — target shape, context length "
                "`MIN_CONTEXT`..`MAX_CONTEXT`, horizon 1..`MAX_HORIZON`, finiteness, and the covariate time-axis "
                "contract — and returns an **input manifest** naming the schema and ceilings, each variate's observed "
                "context length and value range, and the verdict. The manifest is written to "
                "`outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates a "
                "deliberately too-short context and records the pipeline's own error message as a finding. The naive "
                "**last-value baseline** (repeat the final observed value across the horizon) is computed here from the "
                "same context."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_CONTEXT': MIN_CONTEXT, 'MAX_CONTEXT': MAX_CONTEXT, 'MAX_HORIZON': MAX_HORIZON}}}})\n"
                "context = series[:, :-HORIZON]\n"
                "truth = series[:, -HORIZON:]\n"
                "input_manifest = validate_inputs(context, horizon=HORIZON, names=variate_names)\n"
                "# Demonstrate rejection on an input that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(np.ones(MIN_CONTEXT - 1), horizon=HORIZON)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'short-context-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "baseline = last_value_baseline(context, HORIZON)\n"
                "print(json.dumps(input_manifest, indent=2))\n"
                "print({{'context_length': context.shape[1], 'horizon': HORIZON, 'baseline_mae': mae(truth, baseline), 'baseline_rmse': rmse(truth, baseline)}})"
            ),
        },
        {
            "md": (
                "## 6. Forecast\n\n"
                "`forecast` returns `quantiles` of shape `(variates, 9, horizon)` at `quantile_levels` 0.1 … 0.9, and "
                "`median` — the q=0.5 slice, which is the **point forecast** (`point_forecast` in the result). The median "
                "is a model median, not a mean, and the other quantiles are model quantiles rather than guaranteed "
                "confidence intervals; nothing is calibrated here. The horizon and the effective context length are "
                "echoed in the result. Inference is zero-shot and deterministic given the same weights, device and "
                "library versions; CPU and CUDA kernels may differ at floating-point precision. Look for the first "
                "steps of the median next to the withheld truth."
            ),
            "code": (
                "result = pipe.forecast(context, horizon=HORIZON)\n"
                "prediction = result['median']\n"
                "print({{'point_forecast': result['point_forecast'], 'quantile_levels': list(result['quantile_levels']), 'horizon': result['horizon'], 'context_length': result['context_length'], 'n_variates': result['n_variates'], 'device': result['device'], 'source': result['source']}})\n"
                "for step in range(min(5, HORIZON)):\n"
                "    print(f\"step {{step + 1:>2}}  median {{prediction[0, step]:.4f}}  truth {{truth[0, step]:.4f}}  q10 {{result['quantiles'][0, 0, step]:.4f}}  q90 {{result['quantiles'][0, 8, step]:.4f}}\")"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. Because the "
                "truth was withheld chronologically in Section 5, it carries the repository's own `mae` and `rmse` on "
                "the median, the `interval_coverage` of the q10–q90 band (nominal 0.8), and the same `mae`/`rmse` for "
                "the `last_value_baseline`, with the verdict `sample-sanity` — one holdout on the tutorial sample with no "
                "dispersion estimate, not a benchmark. Without withheld truth the verdict is `not-measurable` and the "
                "report states what would make the task measurable (a chronological holdout repeated over representative "
                "periods). The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, truth, context=context, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No withheld truth was supplied, so mae/rmse are not computed; the forecast above is sanity evidence only.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the forecast summary (point-forecast semantics, quantile levels, horizon, "
                "context length), the evaluation report, the input manifest, the sample identity and digest, the "
                "notebook's source (repository, revision, embedded module digest, generator), the model identifier, the "
                "immutable model revision, and the runtime identity (Python, `torch`, `numpy`, `pandas`, device). The "
                "CSV keeps time step, variate, median, truth, last-value baseline and all nine quantiles aligned row by "
                "row. No credentials are recorded."
            ),
            "code": (
                "rows = []\n"
                "for variate in range(prediction.shape[0]):\n"
                "    for step in range(HORIZON):\n"
                "        row = {{'variate': variate_names[variate], 'step': step + 1, 'median': float(prediction[variate, step]), 'truth': float(truth[variate, step]), 'last_value_baseline': float(baseline[variate, step])}}\n"
                "        for index, level in enumerate(result['quantile_levels']):\n"
                "            row[f'q{{int(round(level * 100)):02d}}'] = float(result['quantiles'][variate, index, step])\n"
                "        rows.append(row)\n"
                "pd.DataFrame(rows).to_csv('outputs/{stem}_forecast.csv', index=False)\n"
                "payload = {{\n"
                "    'forecast': {{'point_forecast': result['point_forecast'], 'quantile_levels': list(result['quantile_levels']), 'horizon': result['horizon'], 'context_length': result['context_length'], 'n_variates': result['n_variates'], 'median': prediction.tolist()}},\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': sample_name, 'shape': list(series.shape), 'float32_sha256': sample_sha256}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'numpy': numpy.__version__,\n"
                "        'pandas': pandas.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The forecast is zero-shot; no gradient training or fine-tuning occurs. q=0.5 is a model median, and the "
        "other quantiles are model quantiles rather than guaranteed confidence intervals — the reported q10–q90 "
        "coverage on one holdout is not a calibration statement. MAE/RMSE come from one chronological tutorial holdout "
        "of a synthetic series and must be repeated over representative periods of the real deployment series before "
        "any conclusion; the last-value baseline is the floor a useful forecaster must beat on that series, not a "
        "benchmark. Regime changes, missing values (rejected by the contract), irregular sampling and horizons far "
        "beyond the context all degrade results in ways the pipeline does not detect. The pipeline provides no "
        "fine-tuning, online adaptation, anomaly detection or imputation capability.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can "
        "acquire and digest-verify the pinned checkpoint, validate the demonstrated input, execute the public "
        "pipeline path, and emit the shown machine-readable outputs in the tested runtime — without the repository "
        "being reachable. It does **not** establish benchmark superiority, deployment calibration, safety for "
        "high-consequence decisions, or production fitness on an unseen domain.\n\n"
        "**Next experiments:** enable `USE_BYOD` with a CSV from your own domain and compare the median's MAE against "
        "the last-value baseline over several consecutive holdouts (roll `HORIZON` forward); pass `past_covariates` / "
        "`future_covariates` through `pipe.forecast` to see the covariate time-axis contract enforced; shorten the "
        "context toward `MIN_CONTEXT` and watch the q10–q90 band widen.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/tirex-forecasting-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/tirex-forecasting-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/tirex-forecasting-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/NX-AI/tirex-2\n"
        "- Paper: https://arxiv.org/abs/2607.01204"
    ),
}
