# Weight provenance and DIMER hosting

- Upstream: `NX-AI/TiRex-2`
- Immutable revision: `05e5b26db52bfb256f1ae1bdf785589850482de3`
- Upstream package: `tirex-2==0.2.1`
- Weight format: `model.ckpt` PyTorch checkpoint
- Upstream license: Apache-2.0
- DIMER hosting: Apache-2.0 permits redistribution and hosted use subject to preservation of license/notice obligations.
- Deserialization boundary: upstream TiRex-2 loads the checkpoint with `torch.load(..., weights_only=True)`. It remains a PyTorch checkpoint rather than SafeTensors, so DIMER should retain the immutable-revision and trusted-source requirement and should not accept arbitrary user-supplied checkpoint substitution.
