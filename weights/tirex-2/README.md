---
datasets:
- autogluon/chronos_datasets
- Salesforce/lotsa_data
pipeline_tag: time-series-forecasting
library_name: tirex-2
license: apache-2.0
---
<div align="left" class="flex items-baseline gap-2">
  <img src="tirex.svg" alt="TiRex mascot" width="33" height="40" />
  <h1 class="m-0">TiRex-2: Generalizing TiRex to Multivariate Data and Streaming</h1>
</div>

This repository provides the pretrained TiRex-2 model and inference code for zero-shot
 multivariate forecasting with past and future-known covariates, as introduced in [TiRex-2:
 Generalizing TiRex to Multivariate Data and Streaming](https://arxiv.org/abs/2607.01204).

TiRex-2 is a pretrained time series foundation model that forecasts one or many target
variates directly from their history, optionally conditioned on past and future-known
covariates. A single checkpoint serves both univariate and multivariate forecasting and
operates in a streaming fashion as new observations arrive — all zero-shot, with no
task-specific training or fine-tuning.

TiRex-2 generalizes our original univariate model, [TiRex](https://huggingface.co/NX-AI/TiRex), to
multivariate forecasting with past and future covariates.

## Key facts

- **Zero-shot multivariate forecasting**: 
  TiRex-2 forecasts multiple target variates out of the box, without training or fine-tuning on you data.

- **Past and future-known covariates**:
  TiRex-2 natively conditions on past covariates and future-known covariates, such as
  calendar features, holidays, promotions, or scheduled interventions.

- **Small active footprint**:
  TiRex-2 activates 38.4M parameters in univariate mode and an additional 44.1M parameters
  for multivariate forecasting.

## Getting started

> 📖 For a detailed guide — including pip installation, a Google Colab demo, covariate
> examples, and benchmark reproduction — see our [GitHub repository](https://github.com/NX-AI/tirex-2).

The environment is managed by [Pixi](https://pixi.prefix.dev/latest/). Run the following to install it on your machine

```bash
curl -fsSL https://pixi.sh/install.sh | sh
git clone https://github.com/NX-AI/tirex-2 && cd tirex-2
# activate the cpu-only env
eval "$(pixi shell-hook -e example)"  # to execute on GPUs use `example-cu128` or `example-cu126`
```

Minimal usage predicting a simple sine wave:
```python
import matplotlib.pyplot as plt, torch
from tirex2 import TimeseriesType, load_model

model = load_model("NX-AI/TiRex-2", device="cpu")  # use `device="cuda"` if cuda is available
y = torch.sin(torch.arange(160).float() / 8) 
ts = TimeseriesType(target=y[:128].unsqueeze(0), past_covariates=None, future_covariates=None)
forecast = model.forecast([ts], prediction_length=32, output_type="numpy")[0][0]
```

We provide predefined Pixi tasks showcasing examplary forecasts. These run in the CPU-only `example` environment by default:
- `pixi run minimal` runs above code and creates a plot of the forecast.
- `pixi run comparison` showcases the additional benefit of future known covariates in forecasting a target.

For a more **interactive demo of TiRex-2**, we also provide a [quick-start](https://github.com/NX-AI/tirex-2/blob/main/examples/getting_started.ipynb) notebook. 

## TiRex-2 Pro
TiRex-2 already provides state-of-the-art performance for zero-shot prediction, so you can use this open-source release without training on your own data.

Our pro version extends TiRex-2 with additional capabilities, including:

- **Streaming**: incremental forecast updates as new observations arrive, without recomputing over the full history.
- **Speed**: performance-optimized inference, including optimization for dedicated hardware such as edge, embedded, and industrial PC deployments.
- **Finetuning**: models fine-tuned on your data or with different pretraining.
- **Classification & Regression**: TiRex-2 adapted for classification and regression tasks.

If you are interested in any of these, please contact us at [contact@nx-ai.com](mailto:contact@nx-ai.com).

## Cite
If you use TiRex-2 in your research, please cite our work:
```bibtex
@misc{podest2026tirex2generalizingtirexmultivariate,
      title={TiRex-2: Generalizing TiRex to Multivariate Data and Streaming}, 
      author={Patrick Podest and Marco Pichler and Elias Bürger and Levente Zólyomi and Bernhard Voggenberger and Wilhelm Berghammer and Daniel Klotz and Sebastian Böck and Günter Klambauer and Sepp Hochreiter},
      year={2026},
      eprint={2607.01204},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2607.01204}, 
}
```

## Other versions:
Alongside this pretrained checkpoint, we release decontaminated versions to enable fair 
zero-shot evaluation on specific benchmarks by excluding their data from pretraining:
- [TiRex-2-g](https://huggingface.co/NX-AI/TiRex-2-gifteval-zs): excludes any overlap with 
  the GiftEval datasets ([pretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain) 
  and [evaluation](https://huggingface.co/datasets/Salesforce/GiftEval)) from pretraining.
- [TiRex-2-gp](https://huggingface.co/NX-AI/TiRex-2-gifteval-pretrain): includes the 
  [GiftEval-Pretrain collection](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain) 
  in the pretraining corpus (for comparison against TiRex-2-g).
- [TiRex-2-f](https://huggingface.co/NX-AI/TiRex-2-fevbench): excludes all 
  [fev-bench eval datasets](https://huggingface.co/datasets/autogluon/fev_datasets) 
  from pretraining, using the same approach as for GiftEval.