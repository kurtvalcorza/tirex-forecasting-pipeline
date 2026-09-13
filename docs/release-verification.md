# Release verification

`tutorials/tirex_forecasting_colab.ipynb` (`TASK-INFERENCE`, **standalone** carrier) is a **release
candidate** until the exact notebook revision has executed top-to-bottom in a clean supported runtime.
Unit tests, JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 1.1. This file is the
durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec `1.1`,
  `standalone: true` and `generated_from` (repository, generating revision, the carried modules, their concatenated
  SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the
  primary path; one cell tagged `embedded_module` per carried module (`src/tirex_forecasting_pipeline/evaluation.py`,
  `validation.py`, `pipeline.py`, in dependency order), each equal to the module after the generator's documented
  rewrites (working-directory-relative weights directory; package-relative imports removed); the inline `MANIFEST`
  equal to the committed snapshot manifest and the inline `PINS` equal to the `pyproject.toml` runtime pins; the
  notebook byte-identical to `tools/build_notebook.py` output; the pinned-install cell with its restart-on-stale-import
  guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cells (and repeated in the inline manifest,
  which the notebook asserts against the module before fetching), the revision is a 40-hex immutable commit, and the
  same identity string appears in `README.md`, `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `TiRexForecastPipeline.from_pretrained(weights_dir=...)`, `validate_inputs`, `forecast`, `evaluation_report`,
  `last_value_baseline`), the ceiling print (`MIN_CONTEXT`, `MAX_CONTEXT`, `MAX_HORIZON`), the chronological holdout
  (`context = series[:, :-HORIZON]`, `truth = series[:, -HORIZON:]`), the exports, the learner-facing forecasting
  statements (median semantics, model quantiles are not confidence intervals, last-value baseline, no fine-tuning)
  and the gated-off BYOD default listed in the validator; forbidden patterns (credential-in-URL, any `git clone` /
  `github.com` / repository import on the primary path, a mutable `revision='main'`, direct `tirex2` /
  `huggingface_hub` use **outside the carried module cells**, any worker process or subprocess outside the
  generator-owned install cell, `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

CI also runs `ruff`, `tools/build_notebook.py --check`, and the offline unit suite (`tests/test_pipeline.py`,
`tests/test_snapshot.py`, `tests/test_role_helpers.py`, `tests/test_notebook_parity.py`, `tests/test_release_assets.py`;
stubbed `tirex2`, no weights). These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebook is standalone) |
| Kaggle CLI kernel, fresh-interpreter harness | Same Kaggle container; the committed notebook is executed verbatim, cell by cell, by `run_nb.py` in a subprocess of the container Python | Used when the kernel pre-imports a distribution the pinned install replaces (numpy 2.0.2 vs the pinned 2.3.3): the stale-import guard correctly halts the in-kernel path, so the verbatim notebook runs in a fresh interpreter instead; the evidence cell proves the executed file equals the committed blob |
| Local WSL harness (pre-flight only) | Workstation, `run_nb.py` sequential cell executor with a `google.colab` shim | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU runtime (Colab, or the Kaggle executor above) with
   **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS` (= `pyproject.toml`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (define `TiRexForecastPipeline`, `validate_inputs`, `evaluation_report`,
     the metric helpers and the ceilings) with no import of the repository package;
   - pinned `NX-AI/TiRex-2` acquisition at the immutable revision through the package: the inline `MANIFEST` is
     asserted against the module identity and written to `weights/tirex-2/`, `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reports all three manifest entries (`README.md`, `model-config.yaml`, `model.ckpt`) on a
     clean runtime, `verify_snapshot` returns the manifest dict, and `from_pretrained(weights_dir=WEIGHTS_DIR)`
     reports `source == 'local-snapshot'`;
   - deterministic synthetic sample (256 steps, seed 7) with its float32 SHA-256 printed, the ceilings surfaced,
     the final 32 steps withheld chronologically and the last-value baseline computed from the context;
   - `validate_inputs` writes `outputs/tirex_forecasting_input_manifest.json` (verdict `accepted`, one recorded
     rejection finding from the short-context probe);
   - zero-shot forecast through `forecast(context, horizon=HORIZON)` with q=0.1–0.9 outputs, `point_forecast ==
     'median (q=0.5)'`, context 224 / horizon 32;
   - `evaluation_report` writes `outputs/tirex_forecasting_evaluation_report.json` with verdict `sample-sanity`
     carrying `mae`, `rmse`, `interval_coverage` and the `last_value_baseline` comparison (the previous notebook's
     runs recorded MAE 0.049102 / RMSE 0.064577 vs baseline 0.672854 / 0.746826 on the same sample and holdout;
     the standalone path must be measured, not assumed to reproduce them);
   - `outputs/tirex_forecasting_result.json` and `outputs/tirex_forecasting_forecast.csv` written with
     `NOTEBOOK_SOURCE`, model revision, model licence, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, tirex-2, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/tirex_forecasting_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/tirex_forecasting_colab.ipynb`). Wall times are the sum of per-cell
times reported by the executor and include installs and the model download; they are
measurements for the stated runtime, not general estimates.

### Standalone carrier (NOTEBOOK_SPEC 1.1 §3.6) — current notebook

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| | | | Default sample path | | pending — queued to the GPU lane |

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
