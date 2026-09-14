# Weight provenance and DIMER hosting

- Upstream: `NX-AI/TiRex-2`
- Immutable revision: `05e5b26db52bfb256f1ae1bdf785589850482de3`
- Upstream package: `tirex-2==0.2.1`
- Weight format: `model.ckpt` PyTorch checkpoint (380613375 bytes, SHA-256 `184b160ffbe4c01a26beeba14015ff3507c7497e1f3577114187bbc1d19fcac1`) plus `model-config.yaml`
- Upstream license: Apache-2.0
- Local snapshot: `weights/tirex-2/` with `dimer-base-manifest.json` (per-file bytes + SHA-256 for `README.md`, `model-config.yaml`, `model.ckpt`; `totalBytes` 380620100); the Git repository commits the manifest and the small files and git-ignores the checkpoint.
- Load-time check: `stage_missing_files()` fetches only absent manifest entries at the immutable revision and `verify_snapshot()` in `src/tirex_forecasting_pipeline/pipeline.py` re-hashes every entry and refuses on any mismatch before `tirex2.load_model` reads the directory.
- DIMER hosting: Apache-2.0 permits redistribution and hosted use subject to preservation of license/notice obligations.
- Deserialization boundary: upstream TiRex-2 loads the checkpoint with `weights_only=True` PyTorch state-dict deserialization. It remains a PyTorch checkpoint rather than SafeTensors, so DIMER should retain the immutable-revision and trusted-source requirement (the digest-verified file at the pinned revision is the trust boundary) and should not accept arbitrary user-supplied checkpoint substitution.
