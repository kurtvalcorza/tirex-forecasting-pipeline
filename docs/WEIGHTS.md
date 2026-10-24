# Weight provenance and DIMER hosting

- Upstream: `NX-AI/TiRex-2`
- Immutable revision: `05e5b26db52bfb256f1ae1bdf785589850482de3`
- Upstream package: `tirex-2==0.2.1`
- Weight format: `model.ckpt` PyTorch checkpoint (380613375 bytes, SHA-256 `184b160ffbe4c01a26beeba14015ff3507c7497e1f3577114187bbc1d19fcac1`) plus `model-config.yaml`
- Served file: `model.safetensors` (380555776 bytes, SHA-256 `7a97f73883d8466552afa79347d44f7730ca91dd82aa345606e8d339dbbe0c72`, pinned as `CONVERTED_SHA256`), converted locally from the verified `model.ckpt` by `convert_checkpoint`; never downloaded
- Upstream license: Apache-2.0
- Local snapshot: `weights/tirex-2/` with `dimer-base-manifest.json` (per-file bytes + SHA-256 for `README.md`, `model-config.yaml`, `model.ckpt`; `totalBytes` 380620100); the Git repository commits the manifest and the small files and git-ignores the checkpoint.
- Load-time check: `stage_missing_files()` fetches only absent manifest entries at the immutable revision and `verify_snapshot()` in `src/tirex_forecasting_pipeline/pipeline.py` re-hashes every entry and refuses on any mismatch; `ensure_converted()` then converts `model.ckpt` once (or reuses a `model.safetensors` whose digest equals the pin) and `load_converted_model()` builds `TiRex2` from `model-config.yaml` exactly as `tirex2.load_model` 0.2.1 does and loads the safetensors file with `load_state_dict(strict=True)`.
- DIMER hosting: Apache-2.0 permits redistribution and hosted use subject to preservation of license/notice obligations.
- Deserialization boundary: the published checkpoint is a PyTorch pickle, read exactly once, after its digest check, with `torch.load(weights_only=True)` to convert it; the model is served from the pinned safetensors conversion and the pickle is not deserialised again. The source remains a PyTorch checkpoint rather than SafeTensors, so DIMER should retain the immutable-revision and trusted-source requirement (the digest-verified file at the pinned revision is the trust boundary) and should not accept arbitrary user-supplied checkpoint substitution.
