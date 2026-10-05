# Release verification

`tutorials/tirex_forecasting_colab.ipynb` (`TASK-INFERENCE`, `GUIDED`, **standalone** carrier, generator /3) is a
**release candidate** until the exact notebook revision has executed top-to-bottom in a clean supported runtime with one
**Run all** in the notebook kernel. Unit tests, JSON validation, code-cell compilation, and
`tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER Notebook
Specification 2.2. This file is the durable release-gate record for the notebook. The executions recorded below ran the
previous (/2, in-kernel install) notebook; they do not carry over to the regenerated notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or execution
  counts; no unresolved placeholder markers (including template braces in markdown); every code cell is preceded by an
  explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE` profile, the spec version
  (`2.2`) and the standalone carrier; `metadata.dimer` declares the profile, mode `GUIDED`, `standalone: true` and
  `generated_from` (repository, generating revision, package paths and SHA-256, carried-file digests, generator
  `build_notebook.py/3.0`);
- the standalone carrier and isolated environment (ST1–ST6, PAR1–PAR3, RUN1, RUN10, ENV6): one carrier cell whose
  carried files equal the repository files (`src/tirex_forecasting_pipeline/{__init__,evaluation,pipeline,validation}.py`,
  the stage runner, `tutorials/requirements-colab.lock.txt`, the 3-file snapshot manifest, `LICENSE`) with matching
  digests; the lock pins every `pyproject.toml` runtime pin with hashes; a pinned `uv` builds a managed-CPython
  environment with `--require-hashes`, reused per lock digest; no in-kernel install and no restart instruction; the four
  Infrastructure cells are titled and collapsed; every learner cell runs a stage; the notebook byte-identical to
  `tools/build_notebook.py` output;
- the stage-runner markers (staging and verification, the synthetic generator, the BYOD header, delimiter, numeric,
  timestamp and trim checks, the chronological holdout, `validate_inputs` with the short-context probe, the last-value
  baseline, the least-squares trend + season reference and the noise floor, the CPU-only model load with the GPU hidden,
  `forecast`, `evaluation_report` with the added reference points and coverage granularity, the provenance record), the
  form-parameter defaults (calls that appear only in comments do not count), no quality `assert`, and the forbidden
  patterns (credential-in-URL, any clone or repository import on the primary path, a mutable revision, model-library use
  in the notebook's own cells, `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

CI also runs `ruff`, `tools/build_notebook.py --check`, and the offline unit suite (`tests/`; stubbed `tirex2`, no
weights), including `tests/test_release_assets.py` (negative controls proving the validator discriminates) and
`tests/test_notebook_review_fixes.py` (the notebook's own cells with stand-ins, and the model-free stages). These are
source and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime; any kernel Python — the stages run on the isolated environment's CPython 3.12.12 | The runtime the tutorial is written for; a clean one-pass **Run all** in the notebook kernel is promotion evidence |
| Kaggle notebook kernel | Kaggle CPU kernel; the committed notebook run verbatim with **Run all** (no repository checkout) | Reproducible clean-room executor of the same class; the kernel's preloaded numpy no longer matters, because nothing is installed into the kernel |
| Local harness (pre-flight only) | Workstation, sequential cell executor, stand-ins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new runtime (Colab CPU, or a Kaggle notebook kernel) with **no repository
   checkout** and a clean model cache;
3. choose **Run all** once with the defaults (`USE_BYOD = False`, `HORIZON = 32`, `SEASON_PERIOD = 0.0`,
   `RUN_ACTIVITY = False`); no restart is expected; then re-run the export cell (Section 8) once;
4. verify the carried-file verification and the isolated environment's versions (CPython 3.12.12, `torch` 2.8.0,
   `tirex-2` 0.2.1, `numpy` 2.3.3, `pandas` 2.3.3); the 3-file snapshot staged and verified; the synthetic sample's
   float32 SHA-256 `55e436de…`; context 224 / horizon 32; the input manifest with the short-context refusal; last-value
   MAE 0.6729 / RMSE 0.7468, the least-squares trend + season reference 0.0437 / 0.0576 and the noise floor
   0.0414 / 0.0555; the forecast on the CPU with `point_forecast == 'median (q=0.5)'`; the evaluation report
   (`sample-sanity`, the model's MAE / RMSE — the previous notebook's runs recorded 0.049102 / 0.064577 on the same sample;
   the regenerated path must be measured, not assumed to reproduce them — coverage and its granularity, the
   interpretation line); the CSV and result JSON with source, model identity, licence and runtime;
5. record the notebook Git blob id, commit, executor (**Run all** in the notebook kernel), `restarted: false`, runtime
   (platform, Python, PyTorch, tirex-2, device), model identifier and immutable revision, whether the model cache was
   clean, outcome, metrics and outputs in the table below;
6. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/tirex_forecasting_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/tirex_forecasting_colab.ipynb`). Wall times are the sum of per-cell
times reported by the executor and include installs and the model download; they are
measurements for the stated runtime, not general estimates.

### Standalone carrier (NOTEBOOK_SPEC 1.1 §3.6) — current notebook

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `4863984` / `9af80d86af9c` | Kaggle CPU (`kurtvalcorza/dimer-nb2-tirex-forecasting` v1) | Default sample path | 262.0 s | **PASSED** — 10/10 ok code cells executed cleanly, 8 files, 381 MB staged |

### Previous repository-installing notebook (NOTEBOOK_SPEC 1.0) — audit trail, does not cover the standalone carrier

Pre-flight runtime: WSL2 Ubuntu 24.04 (kernel 6.18.33), Python 3.12.3, Intel Core Ultra 9 275HX (24 threads), 15 GiB RAM, NVIDIA GeForce RTX 5070 Ti Laptop GPU (12,227 MiB, driver 610.88). The harness executes the working-copy notebook cell by cell with the package installed non-editably from the same tree, under an empty `HF_HOME`. **Not a supported user runtime and not promotion evidence.**

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-11 | `b9cdacc5d85e` / blob `537068f5db36` (the notebook blob at this revision; the two later commits change `tools/validate_release_assets.py` lint and documentation only) | Kaggle kernel `kurtvalcorza/dimer-tirex-forecast-verify-v2` v4 — fresh CPU container, Python 3.12.13, Linux 6.12.90; committed notebook executed verbatim, cell by cell, by `run_nb.py` in a fresh interpreter (executed blob == committed blob, measured in-run); the Kaggle kernel itself pre-imports numpy 2.0.2 and Pillow 11.3.0, which the generalized stale-import guard of this revision would reject after the pinned install, hence the fresh interpreter | Default deterministic sample, all 5 code cells, clean HF cache (`models--NX-AI--TiRex-2` only) | 287.9 s (239 s install, 49 s checkpoint fetch + forecast) | **PASS** — `repository_revision` equals the candidate commit; recorded versions torch 2.8.0, tirex-2 0.2.1, numpy 2.3.3, pandas 2.3.3 (pins; the container's torchvision 0.25.0+cpu remains present and orphaned but nothing in this stack imports it); context 224 / horizon 32; MAE 0.049102 / RMSE 0.064577 vs last-value 0.672854 / 0.746826 — identical to every earlier run; `outputs/tirex_forecast.csv` sha256 `38cdac35…d6aab` (byte-identical to the v3 run), `outputs/tirex_provenance.json` sha256 `80b14aa5…ac231` |
| 2026-09-11 | `a45dfef93292` / blob `e567604b20bc` (this revision; executed file verified equal to the committed blob) | Kaggle kernel `kurtvalcorza/dimer-tirex-forecast-verify-v2` v3 — fresh CPU container, Python 3.12.13, Linux 6.12.90; committed notebook executed verbatim, cell by cell, by `run_nb.py` in a fresh interpreter (the Kaggle kernel itself pre-imports numpy 2.0.2, which the tutorial's stale-import guard correctly rejects after the pinned numpy 2.3.3 install — kernel v1 halted there by design) | Default deterministic sample, all 5 code cells, clean HF cache (`models--NX-AI--TiRex-2` is the only cache entry afterwards) | 209.2 s (182 s install, 27 s checkpoint fetch + forecast) | **PASS** — `repository_revision` equals the candidate commit; torch 2.8.0+cu128, tirex-2 0.2.1, numpy 2.3.3, pandas 2.3.3 (pins); context 224 / horizon 32; MAE 0.049102 / RMSE 0.064577 vs last-value baseline 0.672854 / 0.746826 — identical to the local pre-flight to six decimals; `outputs/tirex_forecast.csv` sha256 `38cdac35…d6aab`, `outputs/tirex_provenance.json` sha256 `e5591e84…df071`; only warning: HF unauthenticated-download notice |
| 2026-09-11 | blob `e567604b20bc` (this revision) | Local WSL harness, CPU (`CUDA_VISIBLE_DEVICES=''`), torch 2.8.0+cu128, tirex-2 0.2.1, numpy 2.3.3, pandas 2.3.3 (pins) | Default deterministic sample, all 5 code cells, clean cache | 253.5 s (checkpoint fetch of 364 MB inside the 249.6 s forecast cell) | PASS — `repository_revision` recorded; `NX-AI/TiRex-2` acquired at the pinned revision into an empty cache (`models--NX-AI--TiRex-2` only); context 224 / horizon 32; MAE 0.0491 / RMSE 0.0646 vs last-value baseline 0.6729 / 0.7468; nine quantiles q10–q90 exported; `outputs/tirex_forecast.csv` sha256 `6fc9cc7a…2d17c`, `outputs/tirex_provenance.json` sha256 `92499b26…fe012` |
| 2026-09-11 | pre-fix working tree of `e037914` | Local WSL harness, CPU with the GPU visible and no CUDA toolkit | Default sample path | — | FAILED at cell 9 — `OSError: CUDA_HOME environment variable is not set` raised from `xlstm`'s sLSTM kernel loader, which resolves CUDA include paths at import whenever a GPU is visible even for `device='cpu'`. Supported CPU runtimes have no visible GPU and are unaffected; the pipeline now raises a typed `RuntimeError` naming the fix (`CUDA_VISIBLE_DEVICES=''` or `CUDA_HOME`) and the pre-flight hides the GPU |

## Current status

**No clean-runtime execution of the standalone notebook has been recorded yet**; the run is **pending** and
queued to the GPU lane. The rows above under the previous notebook prove that the pipeline's forecast path,
the pinned checkpoint fetch and the sample/holdout produced stable metrics in a clean Kaggle container, but they
executed the earlier repository-installing carrier: the standalone path (carried module cells, inline manifest,
`stage_missing_files` through `hf_hub_download`, `verify_snapshot` over the real 380 MB checkpoint, and
`tirex2.load_model` on the verified directory) has been validated statically only (parity PASS, carrier probe with
the repository package blocked) and never run. Static validation (`tools/validate_release_assets.py`), nbformat
validation, a `compile()` sweep over every code cell, and the offline unit suite passed on the tutorial source at
the candidate revision, which is necessary but not sufficient. The registry status remains **Candidate** until a
reviewer confirms a recorded run against the notebook blob under review and an integrator promotes it; promotion
is not performed by the builder. Two facts a reviewer should weigh: `stage_missing_files` was exercised only with
an injected downloader in the unit suite (the real `hf_hub_download` fetch of all three manifest entries into a
fresh `weights/tirex-2/` has not been executed), and `from_pretrained(weights_dir=...)` was exercised only with a
stubbed `tirex2.load_model`; the clean run will be the first execution of the standalone path, of the staging path,
and of the local-directory loading path against the real weights.
