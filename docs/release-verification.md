# Release verification

`tutorials/tirex_forecasting_colab.ipynb` (`TASK-INFERENCE`) is a **release candidate** until the exact
notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON
validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 1.0. This file is
the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile and the notebook-spec version; `metadata.dimer` declares that profile and spec `1.0`;
- the fresh-runtime bootstrap (clone by canonical URL, `DIMER_TUTORIAL_REF`, detached checkout of
  the requested revision, restart-on-stale-import guard) and the recorded `REPO_SHA` in exports;
- `MODEL_ID`/`MODEL_REVISION` are imported from the package rather than hard-coded, the revision is
  a 40-hex immutable commit, and the same identity string appears in `README.md`,
  `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls, exports, learner-facing statements and gated-off BYOD
  default listed in the validator; forbidden patterns (credential-in-URL, direct `transformers`
  loading that bypasses the pipeline, `trust_remote_code=True` outside the pipeline boundary,
  `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and immutable provenance.

These are source/provenance checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab`, sets `DIMER_TUTORIAL_REF`, and chdirs to a scratch directory so the bootstrap clones the candidate |
| Local WSL harness (pre-flight only) | Workstation, `run_nb.py` sequential cell executor with a `google.colab` shim | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU runtime (Colab, or the Kaggle
   executor above) with `DIMER_TUTORIAL_REF` set to the candidate commit and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path);
4. verify that Section 1 reports `repository_revision` equal to the candidate commit and that the
   installed core package versions equal the `pyproject.toml` pins;
5. verify every default-path stage completes:
   - fresh bootstrap from GitHub at the candidate revision;
   - pinned `NX-AI/TiRex-2` acquisition at the immutable revision;
   - deterministic synthetic sample, chronological holdout and last-value baseline;
   - zero-shot forecast through `TiRexForecastPipeline.forecast` with q=0.1–0.9 outputs and the q=0.5 median;
   - MAE/RMSE for the model and the baseline;
   - `outputs/tirex_forecast.csv` and `outputs/tirex_provenance.json` written with repository SHA, model revision, runtime versions and device;
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

Pre-flight runtime: WSL2 Ubuntu 24.04 (kernel 6.18.33), Python 3.12.3, Intel Core Ultra 9 275HX (24 threads), 15 GiB RAM, NVIDIA GeForce RTX 5070 Ti Laptop GPU (12,227 MiB, driver 610.88). The harness executes the working-copy notebook cell by cell with the package installed non-editably from the same tree, under an empty `HF_HOME`. **Not a supported user runtime and not promotion evidence.**

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-11 | blob `e567604b20bc` (this revision) | Local WSL harness, CPU (`CUDA_VISIBLE_DEVICES=''`), torch 2.8.0+cu128, tirex-2 0.2.1, numpy 2.3.3, pandas 2.3.3 (pins) | Default deterministic sample, all 5 code cells, clean cache | 253.5 s (checkpoint fetch of 364 MB inside the 249.6 s forecast cell) | PASS — `repository_revision` recorded; `NX-AI/TiRex-2` acquired at the pinned revision into an empty cache (`models--NX-AI--TiRex-2` only); context 224 / horizon 32; MAE 0.0491 / RMSE 0.0646 vs last-value baseline 0.6729 / 0.7468; nine quantiles q10–q90 exported; `outputs/tirex_forecast.csv` sha256 `6fc9cc7a…2d17c`, `outputs/tirex_provenance.json` sha256 `92499b26…fe012` |
| 2026-09-11 | pre-fix working tree of `e037914` | Local WSL harness, CPU with the GPU visible and no CUDA toolkit | Default sample path | — | FAILED at cell 9 — `OSError: CUDA_HOME environment variable is not set` raised from `xlstm`'s sLSTM kernel loader, which resolves CUDA include paths at import whenever a GPU is visible even for `device='cpu'`. Supported CPU runtimes have no visible GPU and are unaffected; the pipeline now raises a typed `RuntimeError` naming the fix (`CUDA_VISIBLE_DEVICES=''` or `CUDA_HOME`) and the pre-flight hides the GPU |

## Current status

Static CI and unit tests are preparatory evidence. The tutorial remains **Candidate** until a clean supported-class CPU run for the exact notebook revision under review is appended to the table and reviewed.
