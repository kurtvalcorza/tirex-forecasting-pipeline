"""Stage runner for the standalone TiRex-2 zero-shot forecasting tutorial (NOTEBOOK_SPEC 2.2 §25.13 isolated environment).

The tutorial notebook carries this file verbatim (as ``tutorial_stages.py`` in its run directory, beside the carried
package under ``src/``) and runs every stage with the interpreter of an isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --outputs OUTPUTS --weights WEIGHTS --stage data --options '{...}'

Nothing is installed into the notebook kernel. Each stage is a separate process and starts from files only: the
verified snapshot under ``--weights``, the series and records of earlier stages under ``RUN_DIR/state`` (NumPy ``.npy``
and JSON), and the learner-facing exports under ``--outputs``. The model runs on the CPU, the reference path: model
stages hide any GPU (``CUDA_VISIBLE_DEVICES=''``) before importing torch, because upstream xlstm otherwise needs a CUDA
toolkit at import even for CPU inference. On failure a stage writes ``RUN_DIR/state/<stage>.error.json``, which the
notebook re-raises in the kernel.

Stages: weights → data → validate → forecast → evaluate → export, plus the optional ``activity``. ``data`` and
``validate`` import no model library, so CI exercises them directly. No stage asserts a quality level.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

STEM = "tirex_forecasting"
PACKAGE = "tirex_forecasting_pipeline"
SYNTHETIC_STEPS = 256
SYNTHETIC_SEED = 7
SYNTHETIC_PERIOD = 2 * math.pi * 8  # sin(t / 8): the generator's own season, in steps
NOISE_SD = 0.05
DELIMITERS = (",", ";", "\t")


class Run:
    """Paths of one run: carried sources and state under ``root``; learner-facing files under ``outputs``."""

    def __init__(self, root: Path, outputs: Path, weights: Path, options: dict[str, Any]) -> None:
        self.root = root
        self.out = outputs
        self.weights = weights
        self.options = options
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    def write_state(self, name: str, value: Any) -> Path:
        path = self.state / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook in order from Section 4)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_output(self, name: str, value: Any) -> Path:
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path


def package(root: Path):
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module(PACKAGE)


def pipeline_module(root: Path):
    package(root)
    return importlib.import_module(f"{PACKAGE}.pipeline")


def snapshot_dir(run: Run, P) -> Path:
    return run.weights / P.MODEL_KEY


def rounded(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits) if math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [rounded(v, digits) for v in value]
    return value


def load_series(run: Run, needed_by: str):
    import numpy as np

    data = run.read_state("data.json", needed_by)
    return data, np.load(run.state / "series.npy")


# --------------------------------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------------------------------


def stage_weights(run: Run) -> None:
    P = package(run.root)
    M = pipeline_module(run.root)
    snapshot = snapshot_dir(run, P)
    snapshot.mkdir(parents=True, exist_ok=True)
    carried = run.root / "weights" / P.MODEL_KEY / M.MANIFEST_NAME
    manifest = json.loads(carried.read_text(encoding="utf-8"))
    if (manifest["modelId"], manifest["revision"]) != (P.MODEL_ID, P.MODEL_REVISION):
        raise RuntimeError("the carried manifest does not name the identity carried by the package; regenerate the notebook")
    shutil.copyfile(carried, snapshot / M.MANIFEST_NAME)
    print({"model_id": P.MODEL_ID, "revision": P.MODEL_REVISION, "license": P.MODEL_LICENSE, "files": len(manifest["files"]), "total_bytes": manifest["totalBytes"]})
    fetched = P.stage_missing_files(snapshot, allow_download=True)
    print({"weights_dir": str(snapshot), "fetched": fetched})
    verified = P.verify_snapshot(snapshot)
    print({"verified": verified, "checkpoint": M.WEIGHTS_FILE, "note": "model.ckpt is a PyTorch checkpoint (a pickle container); tirex2 loads it with weights_only, and it is accepted only at the pinned size and SHA-256"})
    run.write_state("weights.json", {"snapshot": str(snapshot), "fetched": fetched})


# --------------------------------------------------------------------------------------------------
# data (TRX-m2) and validation with baselines (TRX-m1); no model library
# --------------------------------------------------------------------------------------------------


def synthetic_series():
    import numpy as np

    rng = np.random.default_rng(SYNTHETIC_SEED)
    t = np.arange(SYNTHETIC_STEPS)
    signal = 0.02 * t + np.sin(t / 8)
    series = (signal + rng.normal(0, NOISE_SD, len(t)))[None, :]
    return series, signal


def sniff_delimiter(path: Path) -> str:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        first = handle.readline()
    counts = {d: first.count(d) for d in DELIMITERS}
    best = max(counts, key=counts.get)
    return best if counts[best] else ","


def read_byod_csv(P, path: Path, horizon: int) -> dict[str, Any]:
    """Read a BYOD CSV with every refusal naming the file and the rule; trim a long series to the most recent rows."""
    import numpy as np
    import pandas as pd

    if not path.is_file():
        raise FileNotFoundError(f"BYOD_CSV_PATH {str(path)!r} is not a file in this runtime: upload the CSV or correct the path.")
    delimiter = sniff_delimiter(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle, delimiter=delimiter), [])
    header = [h.strip() for h in header]
    if not header or any(not h for h in header) or len(header) != len(set(header)):
        raise ValueError(f"{path.name}: the header must have non-empty, unique column names (duplicates are refused before pandas could rename them); got {header}")
    if "timestamp" not in header:
        raise ValueError(f"{path.name}: no `timestamp` column in the header {header} (delimiter read as {delimiter!r}; comma, semicolon and tab are recognised)")
    names = [c for c in header if c != "timestamp"]
    if not names:
        raise ValueError(f"{path.name}: no target column besides `timestamp`")
    frame = pd.read_csv(path, sep=delimiter, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    frame.columns = [c.strip() for c in frame.columns]
    bad = {}
    missing = {}
    for column in names:
        raw = frame[column].str.strip()
        numbers = pd.to_numeric(raw.where(raw != ""), errors="coerce")
        text = raw[(raw != "") & numbers.isna()]
        if len(text):
            bad[column] = text.unique().tolist()[:3]
        if (raw == "").any():
            missing[column] = [int(i) + 2 for i in raw.index[raw == ""][:3]]  # +2: header line, 1-based
    if bad:
        raise ValueError(f"{path.name}: target columns must be numeric; non-numeric values in {bad}. Remove text columns (or move identifiers out of the file) and fix the values.")
    if missing:
        raise ValueError(f"{path.name}: empty values in {missing} (CSV line numbers); this pipeline does not impute — fill or drop those rows so the series stays regularly spaced.")
    try:
        timestamps = pd.to_datetime(frame["timestamp"].str.strip(), errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path.name}: the `timestamp` column does not parse as dates/times ({exc})") from None
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError(f"{path.name}: timestamps must be unique and strictly increasing")
    steps = timestamps.diff().dropna()
    if len(steps) and steps.nunique() != 1:
        raise ValueError(f"{path.name}: timestamps must be regularly spaced (found {steps.nunique()} different gaps, e.g. {sorted(map(str, steps.unique()))[:3]}); fill or resample the gaps first")
    series = frame[names].astype(float).to_numpy().T
    keep = P.MAX_CONTEXT + horizon
    trimmed = None
    if series.shape[1] > keep:
        trimmed = {"original_rows": int(series.shape[1]), "kept_rows": keep, "dropped_oldest": int(series.shape[1] - keep), "first_kept_timestamp": str(timestamps.iloc[-keep])}
        series = series[:, -keep:]
        timestamps = timestamps.iloc[-keep:]
    if series.shape[1] < P.MIN_CONTEXT + horizon:
        raise ValueError(f"{path.name}: {series.shape[1]} rows; this tutorial needs at least {P.MIN_CONTEXT + horizon} ({P.MIN_CONTEXT} context + {horizon} holdout)")
    values = P.validate_target(series, max_context=keep)
    return {"series": np.asarray(values, dtype=float), "names": names, "delimiter": delimiter, "trimmed": trimmed, "first": str(timestamps.iloc[0]), "last": str(timestamps.iloc[-1]), "step": str(steps.iloc[0]) if len(steps) else None}


def stage_data(run: Run) -> None:
    import numpy as np

    P = package(run.root)
    opts = run.options
    use_byod = bool(opts.get("use_byod", False))
    horizon = P.validate_horizon(int(opts.get("horizon", 32)))
    for path in sorted(run.out.glob(f"{STEM}_*")):
        if path.is_file():
            path.unlink()
    for name in ("data.json", "validated.json", "forecast.json", "evaluate.json"):
        (run.state / name).unlink(missing_ok=True)
    if use_byod:
        path = Path(opts.get("byod_csv_path") or "")
        info = read_byod_csv(P, path, horizon)
        series, names = info["series"], info["names"]
        record = {"sample_kind": "BYOD", "name": path.name, "delimiter": info["delimiter"], "trimmed": info["trimmed"], "time_range": [info["first"], info["last"]], "step": info["step"]}
        if info["trimmed"]:
            print({"trimmed": info["trimmed"], "note": f"kept the most recent MAX_CONTEXT + HORIZON = {info['trimmed']['kept_rows']} rows; older rows cannot be used as context"})
    else:
        series, _signal = synthetic_series()
        names = ["synthetic-trend-season"]
        record = {"sample_kind": "synthetic", "name": "synthetic_trend_season_256.csv", "generator": "0.02·t + sin(t/8) + N(0, 0.05), seed 7"}
    if series.shape[1] < P.MIN_CONTEXT + horizon:
        raise ValueError(f"this tutorial needs at least {P.MIN_CONTEXT + horizon} rows ({P.MIN_CONTEXT} context + {horizon} holdout); got {series.shape[1]}")
    digest = hashlib.sha256(np.ascontiguousarray(series, dtype=np.float32).tobytes()).hexdigest()
    np.save(run.state / "series.npy", series)
    run.write_state("data.json", {**record, "names": names, "horizon": horizon, "shape": list(series.shape), "float32_sha256": digest})
    print({**{k: v for k, v in record.items() if k != "trimmed"}, "shape": list(series.shape), "horizon": horizon, "float32_sha256": digest})


def estimate_period(context) -> tuple[float | None, str]:
    """The dominant period of the linearly detrended context: the largest periodogram peak (3 steps up to half the context),
    refined by the least-squares fit between the neighbouring frequency bins (the FFT grid alone drifts in phase over a
    long context)."""
    import numpy as np

    y = np.asarray(context, dtype=float)
    n = len(y)
    t = np.arange(n, dtype=float)
    X = np.stack([np.ones_like(t), t], 1)
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    power = np.abs(np.fft.rfft(resid)) ** 2
    k_min = 2  # at most half the context per period
    k_max = n // 3  # at least 3 steps per period
    if k_max < k_min:
        return None, "context too short to estimate a season"
    k = int(np.argmax(power[k_min : k_max + 1])) + k_min

    def sse(period: float) -> float:
        D = np.stack([np.ones_like(t), t, np.sin(2 * np.pi * t / period), np.cos(2 * np.pi * t / period)], 1)
        r = y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
        return float(r @ r)

    candidates = np.linspace(n / (k + 1), n / max(k - 1, 1), 201)
    best = float(min(candidates, key=sse))
    return best, f"largest periodogram peak of the detrended context (frequency {k}/{n}), refined by least squares"


def least_squares_forecast(context, horizon: int, period: float | None):
    """Fit level + trend (+ one sine/cosine pair at ``period``) to each variate's context by least squares and extend it."""
    import numpy as np

    values = np.atleast_2d(np.asarray(context, dtype=float))
    n = values.shape[1]
    t_fit = np.arange(n, dtype=float)
    t_new = np.arange(n, n + horizon, dtype=float)

    def design(t):
        columns = [np.ones_like(t), t]
        if period:
            columns += [np.sin(2 * np.pi * t / period), np.cos(2 * np.pi * t / period)]
        return np.stack(columns, 1)

    out = np.empty((values.shape[0], horizon))
    for i, y in enumerate(values):
        coef = np.linalg.lstsq(design(t_fit), y, rcond=None)[0]
        out[i] = design(t_new) @ coef
    return out


def stage_validate(run: Run) -> None:
    import numpy as np

    P = package(run.root)
    data, series = load_series(run, "validate")
    horizon = data["horizon"]
    context, truth = series[:, :-horizon], series[:, -horizon:]
    print({"ceilings": {"MIN_CONTEXT": P.MIN_CONTEXT, "MAX_CONTEXT": P.MAX_CONTEXT, "MAX_HORIZON": P.MAX_HORIZON}})
    manifest = P.validate_inputs(context, horizon=horizon, names=data["names"])
    try:
        P.validate_inputs(np.ones(P.MIN_CONTEXT - 1), horizon=horizon)
    except ValueError as exc:
        manifest["findings"].append({"input": "short-context-probe", "verdict": "rejected", "message": str(exc)})
    if data.get("trimmed"):
        manifest["findings"].append({"input": data["name"], "verdict": "trimmed", "message": f"kept the most recent {data['trimmed']['kept_rows']} of {data['trimmed']['original_rows']} rows"})
    run.write_output(f"{STEM}_input_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    season = float(run.options.get("season_period") or 0)
    if data["sample_kind"] == "synthetic":
        period, period_source = SYNTHETIC_PERIOD, "the generator's own season (sin(t/8), 2π·8 ≈ 50.27 steps): the right functional family, which is the point of the reference"
    elif season > 0:
        period, period_source = season, "SEASON_PERIOD set in Section 5"
    else:
        estimates = [estimate_period(row) for row in context]
        period, period_source = estimates[0][0], estimates[0][1] + " (first variate; set SEASON_PERIOD to override)"
    last = P.last_value_baseline(context, horizon)
    fit = least_squares_forecast(context, horizon, period)
    baselines = {
        "last_value_baseline": {"mae": P.mae(truth, last), "rmse": P.rmse(truth, last), "what": "repeat the last observed value"},
        "least_squares_trend_season": {"mae": P.mae(truth, fit), "rmse": P.rmse(truth, fit), "what": "level + linear trend + one sine/cosine pair fitted to the context by least squares (4 parameters per variate)", "period_steps": period, "period_source": period_source},
    }
    reference = {}
    if data["sample_kind"] == "synthetic":
        _series, signal = synthetic_series()
        noise = truth - signal[None, -horizon:]
        reference["noise_floor"] = {"mae": float(np.mean(np.abs(noise))), "rmse": float(np.sqrt(np.mean(noise**2))), "what": f"the generator's own N(0, {NOISE_SD}) noise on the holdout: no forecaster can expect to beat it"}
    np.save(run.state / "least_squares.npy", fit)
    run.write_state("validated.json", {"baselines": baselines, "reference": reference})
    for name, row in {**baselines, **reference}.items():
        print({name: rounded({k: v for k, v in row.items() if k in ("mae", "rmse", "period_steps")})})


# --------------------------------------------------------------------------------------------------
# model stages (CPU reference path)
# --------------------------------------------------------------------------------------------------


def cpu_pipeline(run: Run, P):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch is imported: the reference path is the CPU
    return P.TiRexForecastPipeline.from_pretrained(device="cpu", weights_dir=snapshot_dir(run, P), allow_download=False)


def stage_forecast(run: Run) -> None:
    import numpy as np

    P = package(run.root)
    data, series = load_series(run, "forecast")
    horizon = data["horizon"]
    context, truth = series[:, :-horizon], series[:, -horizon:]
    pipe = cpu_pipeline(run, P)
    result = pipe.forecast(context, horizon=horizon)
    np.save(run.state / "quantiles.npy", result["quantiles"])
    summary = {k: result[k] for k in ("point_forecast", "horizon", "context_length", "n_variates", "device", "source")} | {"quantile_levels": list(result["quantile_levels"])}
    run.write_state("forecast.json", summary)
    print(summary)
    q = result["quantiles"]
    for step in range(min(5, horizon)):
        print(f"step {step + 1:>2}  median {q[0, 4, step]:.4f}  truth {truth[0, step]:.4f}  q10 {q[0, 0, step]:.4f}  q90 {q[0, 8, step]:.4f}")


def result_from_state(run: Run, needed_by: str) -> dict[str, Any]:
    import numpy as np

    summary = run.read_state("forecast.json", needed_by)
    quantiles = np.load(run.state / "quantiles.npy")
    return {**summary, "quantiles": quantiles, "median": quantiles[:, summary["quantile_levels"].index(0.5), :]}


def stage_evaluate(run: Run) -> None:
    P = package(run.root)
    data, series = load_series(run, "evaluate")
    validated = run.read_state("validated.json", "evaluate")
    horizon = data["horizon"]
    context, truth = series[:, :-horizon], series[:, -horizon:]
    result = result_from_state(run, "evaluate")
    report = P.evaluation_report(result, truth, context=context, sample_kind=data["sample_kind"])
    ls = validated["baselines"]["least_squares_trend_season"]
    report["baselines"].append({"id": "least_squares_trend_season", "metrics": [{"id": "mae", "value": ls["mae"]}, {"id": "rmse", "value": ls["rmse"]}], "what": ls["what"], "period_steps": ls["period_steps"], "period_source": ls["period_source"]})
    if validated["reference"].get("noise_floor"):
        nf = validated["reference"]["noise_floor"]
        report["reference_points"] = [{"id": "noise_floor", "metrics": [{"id": "mae", "value": nf["mae"]}, {"id": "rmse", "value": nf["rmse"]}], "what": nf["what"]}]
    n_points = truth.size
    report["coverage_granularity"] = f"q10–q90 coverage over {n_points} holdout point(s) moves in steps of 1/{n_points} = {1 / n_points:.4f}"
    model_mae = next(m["value"] for m in report["metrics"] if m["id"] == "mae")
    last_mae = validated["baselines"]["last_value_baseline"]["mae"]
    lines = [f"TiRex median MAE {model_mae:.4f} vs last-value {last_mae:.4f} and least-squares trend + season {ls['mae']:.4f}"]
    if validated["reference"].get("noise_floor"):
        lines.append(f"noise floor {validated['reference']['noise_floor']['mae']:.4f}: on this synthetic series a 4-parameter fit of the right family already sits near the floor, so beating last-value says little about model skill")
    elif model_mae <= ls["mae"]:
        lines.append("TiRex beats the trend + season fit on this holdout; repeat over several holdouts before reading it as skill")
    else:
        lines.append("the trend + season fit does at least as well as TiRex on this holdout; a simple model may suffice for this series")
    report["interpretation"] = "; ".join(lines)
    run.write_output(f"{STEM}_evaluation_report.json", report)
    run.write_state("evaluate.json", {"model_mae": model_mae, "interpretation": report["interpretation"]})
    print(json.dumps(rounded(report), indent=2))


def stage_export(run: Run) -> None:
    import importlib.metadata
    import platform

    import numpy as np
    import pandas as pd

    P = package(run.root)
    data, series = load_series(run, "export")
    horizon = data["horizon"]
    context, truth = series[:, :-horizon], series[:, -horizon:]
    result = result_from_state(run, "export")
    baseline = P.last_value_baseline(context, horizon)
    fit = np.load(run.state / "least_squares.npy")
    rows = []
    for v in range(result["median"].shape[0]):
        for step in range(horizon):
            row = {"variate": data["names"][v], "step": step + 1, "median": float(result["median"][v, step]), "truth": float(truth[v, step]), "last_value_baseline": float(baseline[v, step]), "least_squares_trend_season": float(fit[v, step])}
            for index, level in enumerate(result["quantile_levels"]):
                row[f"q{int(round(level * 100)):02d}"] = float(result["quantiles"][v, index, step])
            rows.append(row)
    pd.DataFrame(rows).to_csv(run.out / f"{STEM}_forecast.csv", index=False)
    source = json.loads((run.root / "source.json").read_text(encoding="utf-8")) if (run.root / "source.json").is_file() else {}
    payload = {
        "forecast": {k: result[k] for k in ("point_forecast", "quantile_levels", "horizon", "context_length", "n_variates")} | {"median": result["median"].tolist()},
        "evaluation_report": json.loads((run.out / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8")),
        "input_manifest": json.loads((run.out / f"{STEM}_input_manifest.json").read_text(encoding="utf-8")),
        "sample": {k: data[k] for k in ("sample_kind", "name", "shape", "float32_sha256")} | {"trimmed": data.get("trimmed")},
        "notebook_source": source,
        "repository_revision": source.get("revision"),
        "model_id": P.MODEL_ID,
        "model_revision": P.MODEL_REVISION,
        "model_license": P.MODEL_LICENSE,
        "runtime": {"python": platform.python_version(), **{d: importlib.metadata.version(d) for d in ("torch", "tirex-2", "numpy", "pandas")}, "device": result["device"]},
    }
    run.write_output(f"{STEM}_result.json", payload)
    print({"rows": len(rows), "columns": list(rows[0]), "runtime": payload["runtime"]})
    print(sorted(p.name for p in run.out.iterdir()))


def stage_activity(run: Run) -> None:
    """TRX-M2: forecast from a shorter context and print MAE and the mean q10–q90 band width beside the full context; writes
    only to outputs/activity/."""
    import numpy as np

    P = package(run.root)
    data, series = load_series(run, "activity")
    horizon = data["horizon"]
    context, truth = series[:, :-horizon], series[:, -horizon:]
    full = result_from_state(run, "activity")
    length = int(run.options.get("context_length", 64))
    if not P.MIN_CONTEXT <= length <= context.shape[1]:
        raise ValueError(f"ACTIVITY_CONTEXT must be between MIN_CONTEXT={P.MIN_CONTEXT} and the full context {context.shape[1]}; got {length}")
    before = sorted(p.name for p in run.out.glob(f"{STEM}_*"))
    pipe = cpu_pipeline(run, P)
    short = pipe.forecast(context[:, -length:], horizon=horizon)

    def summary(result, n):
        q = np.asarray(result["quantiles"])
        return {"context_length": n, "mae": round(P.mae(truth, result["median"]), 4), "mean_q10_q90_width": round(float(np.mean(q[:, 8, :] - q[:, 0, :])), 4), "q10_q90_coverage": round(P.interval_coverage(truth, q[:, 0, :], q[:, 8, :]), 4)}

    rows = [summary(full, context.shape[1]), summary(short, length)]
    for row in rows:
        print(row)
    run.write_output(f"activity/{STEM}_activity_context_{length}.json", {"rows": rows})
    if sorted(p.name for p in run.out.glob(f"{STEM}_*")) != before:
        raise RuntimeError("the activity changed the canonical outputs; it must write only to outputs/activity/")
    print(f"wrote outputs/activity/{STEM}_activity_context_{length}.json (the canonical outputs are unchanged)")


STAGES = {
    "weights": stage_weights,
    "data": stage_data,
    "validate": stage_validate,
    "forecast": stage_forecast,
    "evaluate": stage_evaluate,
    "export": stage_export,
    "activity": stage_activity,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--options", default="{}")
    args = parser.parse_args(argv)
    run = Run(args.root.resolve(), args.outputs.resolve(), args.weights.resolve(), json.loads(args.options))
    error_file = run.state / f"{args.stage}.error.json"
    error_file.unlink(missing_ok=True)
    try:
        STAGES[args.stage](run)
    except BaseException as exc:  # noqa: BLE001 -- every failure is reported to the kernel with its own message
        traceback.print_exc()
        error_file.write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
