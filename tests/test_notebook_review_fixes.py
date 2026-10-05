"""Regression tests for the 2026-10-05 notebook review (TRX-M1, TRX-M2, TRX-m1, TRX-m2, TRX-S1).

They need numpy and pandas (CI's dependencies) but no model: they exec the notebook's own kernel cells with stand-ins for
`run_stage` and `google.colab`, run the stage runner's model-free stages (`data`, `validate`) in-process, and check the
generated notebook statically. Each test names its finding.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
NB = ROOT / "tutorials" / "tirex_forecasting_colab.ipynb"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGES = _load("trx_tutorial_stages", TOOLS / "tutorial_stages.py")


def _nb() -> dict:
    return json.loads(NB.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _md() -> str:
    return "\n".join(_src(c) for c in _nb()["cells"] if c["cell_type"] == "markdown")


def _cell_with(needle: str) -> str:
    found = [_src(c) for c in _nb()["cells"] if c["cell_type"] == "code" and needle in _src(c)]
    assert len(found) == 1, needle
    return found[0]


def _set(src: str, name: str, value) -> str:
    out = []
    for line in src.split("\n"):
        if line.startswith(f"{name} = "):
            line = f"{name} = {value!r}" + (line[line.index("  #"):] if "  #" in line else "")
        out.append(line)
    return "\n".join(out)


@contextlib.contextmanager
def _colab(queue):
    files = types.SimpleNamespace(calls=0)

    def upload():
        files.calls += 1
        return queue.pop(0)

    files.upload = upload
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    saved = {k: sys.modules.get(k) for k in ("google", "google.colab")}
    sys.modules.update({"google": google, "google.colab": colab})
    try:
        yield files
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _kernel(tmp_path: Path) -> tuple[dict, list]:
    calls: list = []
    ns = {"ROOT": tmp_path / "run", "Path": Path, "shutil": shutil, "run_stage": lambda stage, **o: calls.append((stage, o))}
    return ns, calls


def _run(tmp_path: Path, options: dict | None = None):
    root = tmp_path / "run"
    if not (root / "src").exists():
        root.mkdir(parents=True, exist_ok=True)
        (root / "src").symlink_to(ROOT / "src", target_is_directory=True)
    return STAGES.Run(root, tmp_path / "outputs", tmp_path / "weights", options or {})


def _quiet(fn, *args) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fn(*args)
    return out.getvalue()


def _csv(path: Path, rows: int, freq: str = "D", sep: str = ",", extra: dict | None = None) -> Path:
    t = np.arange(rows)
    frame = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=rows, freq=freq).astype(str), "y": 0.01 * t + np.sin(2 * np.pi * t / 12)})
    for name, values in (extra or {}).items():
        frame[name] = values
    frame.to_csv(path, index=False, sep=sep)
    return path


# ---------------------------------------------------------------- TRX-M1: isolated runtime, CPU reference


def test_M1_no_kernel_install_and_no_restart_instruction() -> None:
    text = NB.read_text(encoding="utf-8")
    assert "Restart the runtime" not in text and "restart the runtime" not in text
    own = [_src(c) for c in _nb()["cells"] if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_sources")]
    for src in own:
        assert not re.search(r"['\"]-m['\"]\s*,\s*['\"]pip['\"]|^\s*[%!]\s*pip\b|['\"]pip install", src, re.M), src[:80]
        assert "import torch" not in src and "from tirex_forecasting_pipeline" not in src
    install = _cell_with("# @title Infrastructure: build (or reuse) the isolated")
    assert "'--require-hashes'" in install and "--managed-python" in install
    assert "MPLBACKEND='Agg'" in install and "'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP'" in install


def test_M1_cpu_is_the_reference_path() -> None:
    meta = _nb()["metadata"]
    assert "accelerator" not in meta and "gpuType" not in meta["colab"]
    assert "the CPU is this notebook\\'s reference runtime" in _cell_with("# @title Infrastructure: check the runtime")
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    body = source[source.index("def cpu_pipeline"):source.index("def stage_forecast")]
    assert body.index('os.environ["CUDA_VISIBLE_DEVICES"] = ""') < body.index("from_pretrained(device=\"cpu\"")


def _exec_check_cell(ns: dict, **overrides) -> None:
    src = _cell_with("# @title Infrastructure: check the runtime")
    src = re.sub(r"'environment': [0-9.]+}", "'environment': 0.0}", src, count=1)
    src = re.sub(r"\{'weights': max\(0\.0, [0-9.]+", "{'weights': max(0.0, 0.0", src, count=1)
    for name, value in overrides.items():
        src = _set(src, name, value)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, "<check>", "exec"), ns)


def test_M1_section1_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    _exec_check_cell(ns)
    first = ns["ROOT"]
    (first / "tutorial_stages.py").write_text("# carried")
    _exec_check_cell(ns)
    assert ns["ROOT"] == first and (first / "tutorial_stages.py").is_file()
    _exec_check_cell(ns, NEW_RUN_DIRECTORY=True)
    assert ns["ROOT"] != first


def test_M1_second_run_all_reuses_the_matching_environment(tmp_path, monkeypatch) -> None:
    import subprocess as real_subprocess

    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    _exec_check_cell(ns)
    ns["ENV_ROOT"] = tmp_path / "uvroot"
    ns["NOTEBOOK_SOURCE"] = {"revision": "test"}
    src = _cell_with("# @title Infrastructure: build (or reuse) the isolated")
    version = re.search(r"UV = ENV_ROOT / 'uv-([0-9.]+)'", src).group(1)
    ns["ENV_ROOT"].mkdir()
    (ns["ENV_ROOT"] / f"uv-{version}").write_bytes(b"uv stand-in")
    (ns["ENV_ROOT"] / f"uv-{version}.sha256").write_text(hashlib.sha256(b"uv stand-in").hexdigest())
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append([str(c) for c in cmd])
        if cmd[1] == "venv":
            python = Path(cmd[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("")
        out = json.dumps({"python": "3.12.12", "torch": "x", "tirex-2": "x", "numpy": "x", "pandas": "x", "cuda": False})
        return types.SimpleNamespace(stdout=out + "\n", returncode=0)

    fake = types.SimpleNamespace(run=fake_run, Popen=real_subprocess.Popen, PIPE=real_subprocess.PIPE, STDOUT=real_subprocess.STDOUT)
    for attempt in range(2):
        ns["subprocess"] = fake
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src.replace("import zipfile\n", "import zipfile\nsubprocess = globals()['subprocess']\n", 1), "<install>", "exec"), ns)
        ns["subprocess"] = fake
        assert ns["environment_reused"] is (attempt == 1)
    assert len([c for c in commands if c[1] in ("venv", "pip")]) == 2


# ---------------------------------------------------------------- TRX-M2 / TRX-S1: guided layer, activity, trust boundary

GUIDED = ("## How to use this notebook", "**Who this notebook is for.**", "## The task: Input → Model → Output", "## Roadmap", "<summary><strong>Glossary</strong>", "Predict before running", "**What to notice:**", "Check your reasoning", "## Troubleshooting", "## Conclusion", "Sample conclusion")


def test_M2_guided_layer_and_collapsed_infrastructure() -> None:
    nb = _nb()
    md = _md()
    for heading in GUIDED:
        assert heading in md, heading
    assert "{{" not in md
    infra = [c for c in nb["cells"] if c["cell_type"] == "code" and _src(c).startswith("# @title Infrastructure:")]
    assert len(infra) == 4 and all(c["metadata"].get("cellView") == "form" for c in infra)
    assert "CUDA toolkit" in md  # the documented failure mode has a troubleshooting row


def test_M2_context_activity_writes_only_to_activity() -> None:
    cell = _cell_with("RUN_ACTIVITY = False  # @param")
    assert "ACTIVITY_CONTEXT = 64  # @param" in cell and "run_stage('activity', context_length=ACTIVITY_CONTEXT)" in cell
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    body = source[source.index("def stage_activity"):source.index("STAGES = {")]
    assert '"mean_q10_q90_width"' in body and 'write_output(f"activity/' in body and "!= before" in body


def test_no_quality_assert_in_the_stage_runner() -> None:
    assert not [n for n in ast.walk(ast.parse((TOOLS / "tutorial_stages.py").read_text(encoding="utf-8"))) if isinstance(n, ast.Assert)]


def test_S1_checkpoint_trust_boundary_is_stated() -> None:
    md = _md()
    assert "pickle container" in md and "weights_only" in md


# ---------------------------------------------------------------- TRX-m1: baselines that model trend and season


def test_m1_default_baselines_reproduce_the_review(tmp_path) -> None:
    """TRX-m1: last-value 0.6729 / 0.7468, least-squares trend + season 0.0437 / 0.0576, noise floor 0.0414 / 0.0555."""
    run = _run(tmp_path, {"use_byod": False, "horizon": 32})
    _quiet(STAGES.stage_data, run)
    data = json.loads((run.state / "data.json").read_text())
    assert data["float32_sha256"].startswith("55e436de") and data["shape"] == [1, 256]
    printed = _quiet(STAGES.stage_validate, run)
    validated = json.loads((run.state / "validated.json").read_text())
    b, ref = validated["baselines"], validated["reference"]
    assert (round(b["last_value_baseline"]["mae"], 4), round(b["last_value_baseline"]["rmse"], 4)) == (0.6729, 0.7468)
    assert (round(b["least_squares_trend_season"]["mae"], 4), round(b["least_squares_trend_season"]["rmse"], 4)) == (0.0437, 0.0576)
    assert (round(ref["noise_floor"]["mae"], 4), round(ref["noise_floor"]["rmse"], 4)) == (0.0414, 0.0555)
    assert "short-context-probe" in (run.out / "tirex_forecasting_input_manifest.json").read_text() and "least_squares_trend_season" in printed


def test_m1_evaluation_adds_the_reference_points_and_coverage_granularity() -> None:
    source = (TOOLS / "tutorial_stages.py").read_text(encoding="utf-8")
    body = source[source.index("def stage_evaluate"):source.index("def stage_export")]
    assert '"id": "least_squares_trend_season"' in body and 'report["reference_points"]' in body
    assert "coverage_granularity" in body and 'report["interpretation"]' in body
    md = _md()
    assert "the default series is easy, so the last-value baseline flatters any forecaster" in md and "1/32" in md


def test_m1_period_estimate_and_linear_fallback() -> None:
    t = np.arange(240)
    period, how = STAGES.estimate_period(0.05 * t + np.sin(2 * np.pi * t / 12))
    assert abs(period - 12) < 0.5 and "periodogram" in how
    line = STAGES.least_squares_forecast(np.arange(40, dtype=float)[None, :] * 2.0, 5, None)
    assert np.allclose(line, [[80.0, 82.0, 84.0, 86.0, 88.0]])


# ---------------------------------------------------------------- TRX-m2: BYOD


def test_m2_byod_path_field_and_upload_fallback(tmp_path, monkeypatch) -> None:
    cell = _cell_with("USE_BYOD = False  # @param")
    monkeypatch.setitem(sys.modules, "google.colab", None)
    ns, calls = _kernel(tmp_path)
    exec(compile(_set(_set(_set(cell, "USE_BYOD", True), "BYOD_CSV_PATH", "/d/series.csv"), "HORIZON", 12), "<s4>", "exec"), ns)
    assert calls[0] == ("data", {"use_byod": True, "byod_csv_path": "/d/series.csv", "horizon": 12})
    ns, calls = _kernel(tmp_path)
    with pytest.raises(RuntimeError, match="BYOD_CSV_PATH is empty, and the upload dialog exists only in Google Colab"):
        exec(compile(_set(cell, "USE_BYOD", True), "<s4>", "exec"), ns)
    monkeypatch.undo()
    with _colab([{}]) as files:
        ns, calls = _kernel(tmp_path)
        with pytest.raises(RuntimeError, match="cancelled or empty"):
            exec(compile(_set(cell, "USE_BYOD", True), "<s4>", "exec"), ns)
        assert files.calls == 1 and calls == []


def test_m2_long_series_keeps_the_most_recent_rows_and_reports_it(tmp_path) -> None:
    path = _csv(tmp_path / "hourly.csv", 20_000, freq="h")
    run = _run(tmp_path, {"use_byod": True, "byod_csv_path": str(path), "horizon": 32})
    printed = _quiet(STAGES.stage_data, run)
    data = json.loads((run.state / "data.json").read_text())
    assert data["shape"] == [1, 16_384 + 32] and data["trimmed"]["dropped_oldest"] == 20_000 - 16_416
    assert "kept the most recent" in printed
    _quiet(STAGES.stage_validate, run)
    manifest = json.loads((run.out / "tirex_forecasting_input_manifest.json").read_text())
    assert manifest["inputs"][0]["context_length"] == 16_384 and any(f["verdict"] == "trimmed" for f in manifest["findings"])


def test_m2_semicolon_delimiter_is_read(tmp_path) -> None:
    path = _csv(tmp_path / "semi.csv", 100, sep=";")
    run = _run(tmp_path, {"use_byod": True, "byod_csv_path": str(path), "horizon": 32})
    _quiet(STAGES.stage_data, run)
    assert json.loads((run.state / "data.json").read_text())["delimiter"] == ";"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"site": ["x"] * 100}, r"text\.csv: target columns must be numeric; non-numeric values in \{'site': \['x'\]\}"),
        ({"z": [1.0] * 5 + [""] + [1.0] * 94}, r"text\.csv: empty values in \{'z': \[7\]\}"),
    ],
    ids=["text-column", "empty-value"],
)
def test_m2_bad_columns_are_named(tmp_path, extra, message) -> None:
    path = _csv(tmp_path / "text.csv", 100, extra=extra)
    run = _run(tmp_path, {"use_byod": True, "byod_csv_path": str(path), "horizon": 32})
    with pytest.raises(ValueError, match=message):
        _quiet(STAGES.stage_data, run)


def test_m2_short_and_irregular_series_are_refused(tmp_path) -> None:
    short = _csv(tmp_path / "short.csv", 63)
    with pytest.raises(ValueError, match=r"short\.csv: 63 rows; this tutorial needs at least 64"):
        _quiet(STAGES.stage_data, _run(tmp_path, {"use_byod": True, "byod_csv_path": str(short), "horizon": 32}))
    gap = tmp_path / "gap.csv"
    frame = pd.read_csv(_csv(tmp_path / "full.csv", 100))
    frame.drop(index=10).to_csv(gap, index=False)
    with pytest.raises(ValueError, match=r"gap\.csv: timestamps must be regularly spaced"):
        _quiet(STAGES.stage_data, _run(tmp_path / "b", {"use_byod": True, "byod_csv_path": str(gap), "horizon": 32}))
