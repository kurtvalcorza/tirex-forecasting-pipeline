"""Per-repository template for tools/build_notebook.py /3 (NOTEBOOK_SPEC 2.2 §4 standalone, §25.13 isolated environment).

The generator writes the infrastructure cells (runtime check, carrier, isolated install + stage runner, snapshot staging)
from repository files; this template holds the learner-facing prose, the guided layer and the learner cells. Every
learner cell calls ``run_stage(...)``: the carried ``tools/tutorial_stages.py`` runs one stage per process in an
isolated, hash-locked environment, so nothing is installed into the notebook kernel and no restart is needed. The model
runs on the CPU, the reference path (``cpu_reference``).
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "tirex-forecasting-pipeline"

UV = {
    "version": "0.12.15",
    "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "bytes": 20081404,
    "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
}

TEMPLATE = {
    "package": "tirex_forecasting_pipeline",
    "repo_name": REPO,
    "weights_key": "tirex-2",
    "modules": ["__init__.py", "evaluation.py", "pipeline.py", "validation.py"],
    "entry_module": "pipeline.py",
    "lock": "tutorials/requirements-colab.lock.txt",
    "managed_python": "3.12.12",
    "uv": UV,
    "disk_gib": {"weights": 0.4, "environment": 8.0},
    "runtime_modules": ["torch", "tirex-2", "numpy", "pandas"],
    "install_flags": ["--only-binary", ":all:"],
    "cpu_reference": True,
    "stem": "tirex_forecasting",
    "notebook_name": "tirex_forecasting_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "stage_runner": "tools/tutorial_stages.py",
    "run_all": (
        "Selecting **Run all** in a fresh Linux x86_64 runtime builds an isolated Python environment from the carried "
        "hash-locked requirements without touching the notebook kernel's own packages, stages and digest-verifies the pinned "
        "TiRex-2 checkpoint (~381 MB), generates a deterministic synthetic trend + season series in code, withholds the last "
        "`HORIZON` steps chronologically, validates the context into an input manifest with a recorded rejection, computes a "
        "last-value baseline and a least-squares trend + season reference on the same context, runs the zero-shot forecast on "
        "the CPU, writes an evaluation report comparing the model median with both baselines and the series' noise floor, and "
        "exports the forecast and provenance — without a repository clone, DIMER worker, credential, upload, configuration "
        "edit or runtime restart (NOTEBOOK_SPEC 2.2 §5)."
    ),
    "byod": (
        "Set `USE_BYOD = True` and either set `BYOD_CSV_PATH` to a CSV in this runtime (any runtime) or leave it empty in "
        "Colab to open the upload dialog, then re-run from Section 4. The CSV has a `timestamp` column and one or more numeric "
        "target columns (comma, semicolon or tab separated), with unique, increasing, regularly spaced timestamps and no empty "
        "values, and at least `MIN_CONTEXT + HORIZON` rows. A series longer than `MAX_CONTEXT + HORIZON` rows keeps its most "
        "recent rows, and the trim is reported. Every refusal names the file and the rule (a text column is named with its "
        "values). `HORIZON` is a form field, so you can roll the holdout. Uploaded files stay inside this runtime."
    ),
    "title": "TiRex-2 — DIMER zero-shot forecasting tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            f"https://github.com/kurtvalcorza/{REPO}",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/tirex_forecasting_colab.ipynb",
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
        "At inference TiRex-2 maps a context window of one or more variates (plus optional past and future-known covariates) "
        "to nine forecast quantiles (q=0.1 … q=0.9) per step of the requested horizon in a single forward pass; the pipeline "
        "reports q=0.5 as the point forecast (a model median, not a mean) and keeps all nine quantiles. **No adaptation "
        "occurs:** no gradient training, fine-tuning, streaming adaptation or preprocessing fitting happens in this notebook — "
        "the upstream checkpoint supplies the weights and the model configuration, and the carried package adds snapshot "
        "verification, the input contract (`validate_target`, `validate_horizon`, the covariate time-axis rules), the `mae` / "
        "`rmse` / `last_value_baseline` / `interval_coverage` helpers and the public `validate_inputs` / `evaluation_report` "
        "stages.\n\n"
        "**Read this before Section 7: the default series is easy, so the last-value baseline flatters any forecaster.** The "
        "series is `0.02·t + sin(t/8)` plus small noise. Repeating the last value ignores both the trend and the season (MAE "
        "0.6729), while a 4-parameter least-squares fit of trend + season on the same context reaches 0.0437 — close to the "
        "noise floor of 0.0414 that no forecaster can beat. TiRex's recorded median MAE on this sample is 0.0491. Section 7 "
        "prints all of these side by side, so you can see what the model adds and what it does not; on your own series (BYOD) "
        "the same comparison is the informative one."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to —\n\n"
        "1. **Make** a leakage-safe chronological holdout and **explain** why no future value may reach the model (Section 5).\n"
        "2. **Read** an input manifest and a refusal message from the forecasting contract (Section 5).\n"
        "3. **Interpret** a quantile forecast: the median as the point forecast, the q10–q90 band as model quantiles rather than a guaranteed interval (Section 6).\n"
        "4. **Compare** the model with a last-value baseline, a least-squares trend + season reference and the noise floor, and **say** what the comparison can and cannot show (Section 7).\n"
        "5. **Predict**, run and **explain** how a shorter context changes the error and the band width, in an optional activity (Section 9)."
    ),
    "exclusions": (
        "gradient training or fine-tuning, streaming/online adaptation, classification or regression on tabular features, "
        "anomaly detection, imputation, or calibrated prediction intervals. The quantiles are model quantiles, not guaranteed "
        "coverage; a single holdout is not a deployment-variability estimate."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh **Linux x86_64** runtime — Google Colab, Kaggle or a Linux Jupyter kernel. **The CPU is the reference path**: upstream CUDA execution compiles fused recurrent kernels and needs a CUDA toolkit, so the model stages hide any GPU and run on the CPU. On the CPU, `tirex-2` compiles its residual block with `torch.compile` (TorchInductor) on the first forecast, which needs a C++ compiler (`g++`) in the runtime; Colab and Kaggle images include one. The kernel's own Python version does not matter: the notebook installs nothing into it, and runs every stage with CPython 3.12.12 in an isolated environment built from {n_locked} hash-locked packages (`tirex-2` 0.2.1, `torch` 2.8.0, `numpy` 2.3.3, `pandas` 2.3.3). About 0.4 GB of disk is needed for the checkpoint and about 8 GB for the isolated environment.",
        "- **Knowledge:** basic Python and NumPy. The glossary below explains context window, horizon, quantile and the other forecasting terms.",
        "- **Data:** the default sample is a deterministic synthetic trend + seasonal series (256 steps, fixed seed) generated in code, so nothing is downloaded and no private data is needed. Optional BYOD: one UTF-8 CSV with a unique `timestamp` column and one or more finite numeric target columns in chronological order, regularly spaced, at least 64 rows (32 context + 32 holdout) at the default horizon; longer than 16,416 rows keeps the most recent rows. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "guided": {
        "opening": [
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who can run cells in a hosted notebook and read short Python and NumPy, and "
                "who want to see a time-series foundation model used honestly: a chronological holdout, a quantile forecast read "
                "correctly, and a comparison against baselines that tells you what the model actually adds. No experience with "
                "TiRex or xLSTM is assumed; the glossary below explains every term.\n\n"
                "**Running it.** Choose *Runtime → Run all*. A CPU runtime is enough — it is the reference path. The default path "
                "needs no edit, no upload, no account, no token and no runtime restart. Section 2 builds an isolated environment, "
                "which takes the longest.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell calls "
                "`run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated "
                "environment's Python, streams what it prints, and stops the notebook with the stage's own error message if it "
                "fails. Stages hand results to each other only through files.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–9) are the workflow. *Infrastructure cells* (Sections 1–3) are "
                "collapsed and titled **Infrastructure**; you may run them without studying their implementation.\n\n"
                "**Form controls.** `USE_BYOD`, `BYOD_CSV_PATH` and `HORIZON` (Section 4); `SEASON_PERIOD` (Section 5); "
                "`RUN_ACTIVITY` and `ACTIVITY_CONTEXT` (Section 9). Leave them at their defaults for the first run.\n\n"
                "**Section tags.** **[Concept]** — what the model does and why. **[Evaluation practice]** — how the evidence is "
                "produced and how to read it. **[Engineering]** — reproducibility, provenance and packaging.\n\n"
                "**Predict, then check.** Before Sections 6 and 7 a **Predict before running** prompt asks you to commit to an "
                "expectation; **What to notice** follows each stage; a collapsed **Check your reasoning** answer follows each "
                "checkpoint. The sample answers quote the recorded runs of this sample."
            ),
            (
                "## The task: Input → Model → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Split** | a 256-step series | chronological holdout: the last `HORIZON` = 32 steps withheld | 224 context steps, 32 truth steps |\n"
                "| **Validate** | the context | `validate_inputs` (shape, length 32..16,384, finiteness, covariate axis) | an input manifest with a recorded rejection |\n"
                "| **Baselines** | the context | last value; least-squares level + trend + season | two reference forecasts |\n"
                "| **Forecast** | the context | TiRex-2, zero-shot, on the CPU | nine quantiles per step; the median is the point forecast |\n"
                "| **Evaluate** | forecast, baselines, truth | `evaluation_report` + the trend + season reference + the noise floor | MAE / RMSE, q10–q90 coverage, an interpretation |\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1–3 | [Engineering] | runtime, carried code, isolated environment, verified checkpoint | versions, digests |\n"
                "| 4. Data | [Evaluation practice] | synthetic series or a BYOD CSV | shape, digest, any trim |\n"
                "| 5. Holdout, validation, baselines | [Evaluation practice] | split, input manifest, two baselines | the manifest, baseline errors |\n"
                "| 6. Forecast | [Concept] | zero-shot quantile forecast | median vs truth, the band |\n"
                "| 7. Evaluation | [Evaluation practice] | the report | the comparison and its interpretation |\n"
                "| 8. Export | [Engineering] | CSV and result JSON with provenance | the files |\n"
                "| 9. Optional activity | [Concept] | a shorter context, printed beside the full one | error and band width |\n"
                "| Troubleshooting | [Engineering] | every failure and what to do | when something fails |\n"
                "| Interpretation and conclusion | [Evaluation practice] | what was and was not shown | your conclusion |"
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **Context window** | The past observations the model sees (224 steps by default). |\n"
                "| **Horizon** | How many future steps are forecast (`HORIZON`, 32 by default). |\n"
                "| **Chronological holdout** | Withholding the most recent `HORIZON` steps as truth, so the forecast never sees its own future. |\n"
                "| **Quantile forecast** | For each step, nine values below which the model expects the truth with probability 0.1 … 0.9. |\n"
                "| **Median vs mean** | The q=0.5 quantile is the point forecast here; it is not the average of possible futures. |\n"
                "| **q10–q90 band / coverage** | The range between the 0.1 and 0.9 quantiles; coverage is the share of truth points inside it (nominal 0.8, not guaranteed). |\n"
                "| **Last-value baseline** | Repeat the final observed value across the horizon. |\n"
                "| **Trend + season reference** | A 4-parameter least-squares fit (level, slope, one sine/cosine pair) on the context, extended forward. |\n"
                "| **Noise floor** | The error of a perfect knowledge of the signal: the irreducible noise on the holdout (synthetic series only). |\n"
                "| **Hash-locked environment / stage** | The isolated Python environment every stage runs in; one workflow step run as its own process. |\n\n"
                "</details>"
            ),
        ],
        "infrastructure": {
            "weights": (
                "**Trust boundary (Section 3).** The checkpoint is `model.ckpt`, a PyTorch checkpoint — a pickle container — "
                "because no safetensors file is published at the pinned revision. It is accepted only at the pinned size and "
                "SHA-256, and `tirex2` loads it with `weights_only`, which refuses arbitrary pickled objects. **What to notice:** "
                "`fetched` lists `model.ckpt` on a first run, then the verified-file count."
            ),
        },
    },
    "cells": [
        {
            "md": (
                "## 4. Generate the synthetic sample or supply your own series · [Evaluation practice]\n\n"
                "The default sample is **synthetic**: 256 steps of a linear trend plus a sine season plus small Gaussian noise from "
                "a fixed seed, so it needs no download and its float32 SHA-256 is printed for the record. It has a real future — "
                "the final `HORIZON` steps are withheld in the next section — so the evaluation can score the forecast, but a "
                "synthetic series says nothing about any deployment domain.\n\n"
                "**Bring your own series.** Set `USE_BYOD = True` and `BYOD_CSV_PATH` (or, in Colab, leave it empty for the upload "
                "dialog). The header is inspected before pandas reads the file, so duplicate names cannot be silently renamed; the "
                "delimiter is detected (comma, semicolon or tab); a text value in a target column is refused with the column and "
                "the values; empty values, unparseable or irregular timestamps and too few rows stop with their own message; and a "
                "series longer than `MAX_CONTEXT + HORIZON` keeps its most recent rows and reports the trim."
            ),
            "code": (
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_CSV_PATH = ''  # @param {{type:\"string\"}}\n"
                "HORIZON = 32  # @param {{type:\"integer\"}}\n\n"
                "def upload_one(what, field):\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError:\n"
                "        raise RuntimeError(f'{{field}} is empty, and the upload dialog exists only in Google Colab: set {{field}} to {{what}} in this runtime.') from None\n"
                "    uploaded = files.upload()\n"
                "    if not uploaded:\n"
                "        raise RuntimeError(f'The upload was cancelled or empty: no file was received. Run this cell again and choose {{what}}, or set {{field}}.')\n"
                "    if len(uploaded) != 1:\n"
                "        raise ValueError(f'Upload exactly one file ({{what}}); got {{sorted(uploaded)}}.')\n"
                "    upload_name, payload = next(iter(uploaded.items()))\n"
                "    path = ROOT / 'inputs' / Path(upload_name).name\n"
                "    path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    path.write_bytes(payload)\n"
                "    return str(path)\n\n"
                "csv_path = ''\n"
                "if USE_BYOD:\n"
                "    csv_path = BYOD_CSV_PATH or upload_one('one CSV with a timestamp column', 'BYOD_CSV_PATH')\n"
                "run_stage('data', use_byod=USE_BYOD, byod_csv_path=csv_path, horizon=HORIZON)"
            ),
        },
        {
            "md": "**What to notice:** `sample_kind: 'synthetic'`, shape `[1, 256]`, `horizon: 32` and the float32 SHA-256 `55e436de…`.",
        },
        {
            "md": (
                "## 5. Chronological holdout, validation and baselines · [Evaluation practice]\n\n"
                "The final `HORIZON` steps are **withheld** as the truth; only the earlier context is passed to the model, so no "
                "future value leaks into the forecast. `validate_inputs` applies exactly the checks `forecast` applies — target "
                "shape, context length `MIN_CONTEXT`..`MAX_CONTEXT`, horizon 1..`MAX_HORIZON`, finiteness and the covariate time-axis "
                "contract — and returns an **input manifest**, written to `outputs/{stem}_input_manifest.json`; the stage also "
                "validates a deliberately too-short context and records the refusal as a finding.\n\n"
                "Two baselines are computed from the same context: the **last-value baseline**, and a **least-squares trend + "
                "season reference** — level, slope and one sine/cosine pair fitted to the context. For the synthetic series its "
                "period is the generator's own (2π·8 ≈ 50.3 steps): the right functional family, which is what makes it a fair "
                "reference. For your own series set `SEASON_PERIOD` in steps (for example 7 for daily data with a weekly cycle), or "
                "leave it 0 to use the largest periodogram peak of the detrended context. For the synthetic series the stage also "
                "prints the **noise floor**: the error of knowing the signal exactly."
            ),
            "code": (
                "SEASON_PERIOD = 0.0  # @param {{type:\"number\"}}\n"
                "run_stage('validate', season_period=SEASON_PERIOD)"
            ),
        },
        {
            "md": (
                "**What to notice:** context length 224, the `short-context-probe` refusal, last-value MAE 0.6729 / RMSE 0.7468, the "
                "trend + season reference 0.0437 / 0.0576, and the noise floor 0.0414 / 0.0555.\n\n"
                "**Checkpoint:** why is the trend + season reference fitted on the context only, never on the withheld steps?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Fitting on the withheld steps would let the reference see the answer, exactly the leakage the chronological "
                "holdout exists to prevent. A baseline must play by the same rule as the model: the past only. That is also why "
                "its error is a fair reference point for the model's.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 6. Forecast · [Concept]\n\n"
                "`forecast` returns `quantiles` of shape `(variates, 9, horizon)` at levels 0.1 … 0.9, and `median` — the q=0.5 "
                "slice, which is the **point forecast**. The median is a model median, not a mean, and the other quantiles are model "
                "quantiles rather than guaranteed confidence intervals; nothing is calibrated here. Inference is zero-shot and runs "
                "on the CPU; it is deterministic given the same weights and library versions.\n\n"
                "**Predict before running:** on this smooth series, will the first forecast steps sit close to the truth? Will the "
                "q10–q90 band widen or narrow further into the horizon?"
            ),
            "code": "run_stage('forecast')",
        },
        {
            "md": (
                "**What to notice:** `point_forecast: 'median (q=0.5)'`, `device: 'cpu'`, `source: 'local-snapshot'`, and the first "
                "steps of the median next to the truth, inside the q10–q90 band."
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report · [Evaluation practice]\n\n"
                "`evaluation_report` carries the repository's own `mae` and `rmse` on the median, the `interval_coverage` of the "
                "q10–q90 band (nominal 0.8), and the last-value baseline, with the verdict `sample-sanity` — one holdout, no "
                "dispersion estimate, not a benchmark (without withheld truth the verdict is `not-measurable`). The stage adds the least-squares trend + season reference, the noise floor "
                "(synthetic series), the coverage granularity (with 32 points, coverage moves in steps of 1/32 ≈ 0.031) and an "
                "**interpretation** line. The report is written to `outputs/{stem}_evaluation_report.json`.\n\n"
                "**Predict before running:** TiRex's median versus the trend + season reference — which will have the lower MAE on "
                "this series, and by how much?"
            ),
            "code": "run_stage('evaluate')",
        },
        {
            "md": (
                "**What to notice:** the model's MAE (0.0491 recorded) beside last-value (0.6729), the trend + season reference "
                "(0.0437) and the noise floor (0.0414), the coverage and its granularity, and the interpretation line.\n\n"
                "**Checkpoint:** the model is about 14 times better than last-value. Is that evidence of skill?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Not on this series. Last-value ignores both the trend and the season, so any method that models them looks good "
                "against it. A 4-parameter least-squares fit of the right family reaches 0.0437 — slightly better than the model's "
                "recorded 0.0491 — and both are near the noise floor of 0.0414, so there is almost nothing left to win. TiRex's "
                "result is a good zero-shot forecast with no knowledge of the series' form; whether it adds skill is a question for "
                "real series, compared against a seasonal reference over several holdouts.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance · [Engineering]\n\n"
                "The CSV keeps variate, step, median, truth, the last-value baseline, the trend + season reference and all nine "
                "quantiles aligned row by row. `outputs/{stem}_result.json` preserves the forecast summary, the evaluation report, "
                "the input manifest, the sample identity and digest (with any trim), the notebook's source, the model identifier, "
                "revision and licence, and the isolated environment's versions and device. No credentials are recorded. The stage "
                "only reads earlier results, so re-running it alone is safe."
            ),
            "code": "run_stage('export')",
        },
        {
            "md": (
                "## 9. Optional activity: how much context does the forecast need? · [Concept]\n\n"
                "**Predict → Change → Run → Observe → Explain.** **Predict:** with `ACTIVITY_CONTEXT = 64` instead of the full 224 "
                "steps, will the MAE rise? Will the q10–q90 band widen? **Change:** tick `RUN_ACTIVITY` and set `ACTIVITY_CONTEXT` "
                "(between 32 and 224). **Run** this cell. **Observe** the two rows — full and short context — for MAE, mean band "
                "width and coverage. **Explain** the difference in terms of how many seasonal cycles the model can see. The activity "
                "writes only to `outputs/activity/`."
            ),
            "code": (
                "RUN_ACTIVITY = False  # @param {{type:\"boolean\"}}\n"
                "ACTIVITY_CONTEXT = 64  # @param {{type:\"integer\"}}\n"
                "if RUN_ACTIVITY:\n"
                "    run_stage('activity', context_length=ACTIVITY_CONTEXT)\n"
                "else:\n"
                "    print('Optional activity skipped: tick RUN_ACTIVITY to run it. The canonical outputs are complete.')"
            ),
        },
        {
            "md": (
                "**What to notice (if you ran it):** the `mae` and `mean_q10_q90_width` columns.\n\n"
                "<details>\n<summary>Check your reasoning (open after running)</summary>\n\n"
                "No hosted run of the activity is recorded yet, so compare your rows. 64 steps cover just over one season (about 50 "
                "steps), so the model has less evidence about the period and the trend; expect a larger error and usually a wider "
                "band. A band that widens as evidence shrinks is the quantiles behaving sensibly — but it is still not a calibrated "
                "interval.\n\n"
                "</details>"
            ),
        },
    ],
    "closing": (
        "## Troubleshooting · [Engineering]\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| `This notebook needs a Linux x86_64 runtime` | macOS, Windows or ARM kernel | Use Colab, Kaggle or a Linux x86_64 Jupyter kernel. |\n"
        "| `Not enough free disk` | a used runtime | Start a fresh runtime. |\n"
        "| `uv … wheel size/hash mismatch` or a `--require-hashes` error | a corrupted or substituted download | Re-run Section 2; never remove a pin or a hash. |\n"
        "| a size or SHA-256 mismatch in Section 3 | a file that is not the pinned one | Delete `weights/tirex-2/` and re-run Section 3; never edit the manifest. |\n"
        "| `A CUDA device is visible but no CUDA toolkit was found` | a GPU visible to upstream xlstm | Should not occur: the model stages hide the GPU; report the versions from Section 2 if it does. |\n"
        "| `InvalidCxxCompiler` or `No working C++ compiler found` in Section 6 | a runtime without `g++` (TorchInductor compiles the CPU kernels of `torch.compile` on the first forecast) | Use Colab or Kaggle, or install a C++ compiler in the runtime. |\n"
        "| `Stage '…' failed …: … is missing: run the stage that writes it` | a cell run out of order | Run the notebook in order from Section 4 (or *Run all*). |\n"
        "| `BYOD_CSV_PATH is empty, and the upload dialog exists only in Google Colab` | BYOD outside Colab with no path | Set `BYOD_CSV_PATH`. |\n"
        "| `The upload was cancelled or empty` / `Upload exactly one file` | a cancelled or multi-file upload | Run the cell again and choose one CSV. |\n"
        "| `<file>: no timestamp column in the header` | a missing or misnamed column, or another delimiter | Name the column `timestamp`; comma, semicolon and tab are read. |\n"
        "| `<file>: target columns must be numeric; non-numeric values in {{…}}` | a text or identifier column | Remove the column or fix the values. |\n"
        "| `<file>: empty values in {{…}}` | gaps in a target | Fill or drop those rows; this pipeline does not impute. |\n"
        "| `<file>: timestamps must be regularly spaced` / `unique and strictly increasing` | gaps, duplicates or unsorted rows | Sort, de-duplicate and resample first. |\n"
        "| `<file>: N rows; this tutorial needs at least …` | too short for context + holdout | Supply more rows or lower `HORIZON`. |\n"
        "| `trimmed: {{…}}` (a note, not an error) | more than `MAX_CONTEXT + HORIZON` rows | The most recent rows are kept; older rows cannot be context. |\n\n"
        "## Interpretation and limits\n\n"
        "The forecast is zero-shot; no gradient training or fine-tuning occurs. q=0.5 is a model median, and the other "
        "quantiles are model quantiles rather than guaranteed confidence intervals — the reported q10–q90 coverage on one "
        "holdout of 32 points moves in steps of 0.031 and is not a calibration statement. On the default synthetic series a "
        "4-parameter trend + season fit of the right family is about as accurate as the model and both sit near the noise floor, "
        "so the comparison shows the workflow, not model skill; the last-value baseline alone would overstate what the model adds. "
        "MAE/RMSE from one chronological holdout must be repeated over representative periods of a real deployment series, "
        "against a seasonal reference, before any conclusion. Regime changes, missing values (rejected by the contract), "
        "irregular sampling and horizons far beyond the context all degrade results in ways the pipeline does not detect. The "
        "pipeline provides no fine-tuning, online adaptation, anomaly detection or imputation capability.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this notebook, can acquire and "
        "digest-verify the pinned checkpoint, validate the demonstrated input, execute the public pipeline path, and emit the "
        "shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on "
        "an unseen domain.\n\n"
        "## Conclusion · [Evaluation practice]\n\n"
        "Write three sentences: what the chronological holdout guarantees; how the model's error compares with the last-value "
        "baseline, the trend + season reference and the noise floor on this series; and what you would need before trusting the "
        "model on a real series.\n\n"
        "<details>\n<summary>Sample conclusion (open after writing yours)</summary>\n\n"
        "The holdout withheld the last 32 steps before the model saw anything, so no future value leaked into the forecast or "
        "the baselines. The model's median MAE (0.0491 recorded) is far below last-value (0.6729) but slightly above a 4-parameter "
        "trend + season fit (0.0437), and both are near the noise floor (0.0414), so on this easy series the model shows a "
        "sound zero-shot workflow rather than skill. Before trusting it on a real series I would compare it with a seasonal "
        "reference over several rolling holdouts from representative periods, and check its band coverage over many points.\n\n"
        "</details>\n\n"
        "**Next experiments:** enable `USE_BYOD` with a CSV from your own domain and compare the median's MAE with both baselines "
        "over several consecutive holdouts (lower `HORIZON`, or trim the CSV and re-run); set `SEASON_PERIOD` to your series' "
        "cycle; run the activity at 32, 64 and 128 steps.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/tirex-forecasting-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/tirex-forecasting-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/tirex-forecasting-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/NX-AI/tirex-2\n"
        "- Paper: https://arxiv.org/abs/2607.01204"
    ),
}
