"""Tests for the harness scripts and the static leakage check, on synthetic data with known problems."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

TOOLKIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLKIT))
sys.path.insert(0, str(TOOLKIT / "harness"))

import common  # noqa: E402
import leakage_check  # noqa: E402
import llm_eval  # noqa: E402
import model_diagnostics  # noqa: E402
from dsguard import leakage  # noqa: E402
from dsguard.core import build_context  # noqa: E402

RNG = np.random.default_rng(7)


def test_auc_matches_known_values():
    assert common.auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert common.auc(np.array([0, 1]), np.array([0.5, 0.5])) == pytest.approx(0.5)
    assert np.isnan(common.auc(np.array([1, 1]), np.array([0.2, 0.3])))


def test_calibration_and_psi():
    p = RNG.uniform(size=20_000)
    y = (RNG.uniform(size=p.size) < p).astype(int)
    _, ece_good = common.calibration_table(y, p)
    _, ece_bad = common.calibration_table(y, np.clip(p + 0.2, 0, 1))
    assert ece_good < 0.02 < ece_bad
    a = pd.Series(RNG.normal(size=5000))
    assert common.psi(a, pd.Series(RNG.normal(size=5000))) < 0.05
    assert common.psi(a, pd.Series(RNG.normal(1.5, size=5000))) > 0.25


def _churn_frame(n=4000):
    x = RNG.normal(size=n)
    region = RNG.choice(["north", "south", "east"], size=n)
    logit = 1.5 * x + 0.5 * x**2 - 1                         # model below ignores the x**2 term
    y = (RNG.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    p = 1 / (1 + np.exp(-(1.5 * x - 0.5)))
    p = np.where(region == "east", RNG.uniform(size=n), p)   # model is useless for 'east'
    return pd.DataFrame({"x": x, "noise": RNG.normal(size=n), "region": region, "y": y, "p": p,
                         "leak": y + RNG.normal(0, 0.01, size=n), "id": np.arange(n)})


def test_model_diagnostics_finds_planted_problems(tmp_path):
    data = tmp_path / "eval.csv"
    _churn_frame().to_csv(data, index=False)
    out = tmp_path / "out"
    model_diagnostics.main(["--data", str(data), "--target", "y", "--pred", "p", "--segments", "region",
                            "--id", "id", "--out", str(out)])
    report = json.loads((out / "diagnostics.json").read_text())
    assert report["task"] == "binary"
    rules = {f["rule"] for f in report["flags"]}
    assert {"dominant-feature", "residual-structure", "weak-segment"} <= rules
    assert report["residual_structure"][0]["feature"] in ("x", "leak")
    assert report["segments"][0]["level"] == "east"
    assert (out / "diagnostics.md").read_text().startswith("# Model diagnostics")


def test_model_diagnostics_regression_bias(tmp_path):
    n = 3000
    x = RNG.uniform(0, 10, size=n)
    y = 3 * x + RNG.normal(size=n)
    frame = pd.DataFrame({"x": x, "y": y, "pred": 3 * x - 2})
    data = tmp_path / "reg.csv"
    frame.to_csv(data, index=False)
    model_diagnostics.main(["--data", str(data), "--target", "y", "--pred", "pred", "--out", str(tmp_path)])
    report = json.loads((tmp_path / "diagnostics.json").read_text())
    assert report["task"] == "regression"
    assert report["metrics"]["bias_mean_residual"] == pytest.approx(2, abs=0.1)
    assert "biased" in {f["rule"] for f in report["flags"]}


def test_leakage_check_finds_planted_leaks():
    frame = _churn_frame(3000)
    frame["note"] = ["customer said " + w for w in RNG.choice(["hello", "bye", "thanks a lot"], size=len(frame))]
    frame["event_ts"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(RNG.integers(0, 365, len(frame)), unit="D")
    frame["closed_at"] = frame["event_ts"] + pd.to_timedelta(np.where(frame["y"] == 1, 5, -5), unit="D")
    train, test = frame.iloc[:2000], frame.iloc[1900:]   # 100 shared rows
    result = leakage_check.run_checks(train, test, "y", id_col="id", time_col="event_ts", text_cols=["note"],
                                      exclude=["p"])
    rules = {f["rule"] for f in result["findings"]}
    assert {"shared-id", "duplicate-rows", "duplicate-text", "future-timestamp", "too-predictive"} <= rules
    assert result["single_feature_strength"][0]["feature"] == "leak"


def test_leakage_check_clean_split_is_quiet():
    n = 4000
    frame = pd.DataFrame({"a": RNG.normal(size=n), "b": RNG.normal(size=n)})
    frame["y"] = (frame["a"] + RNG.normal(size=n) > 0).astype(int)
    result = leakage_check.run_checks(frame.iloc[:3000], frame.iloc[3000:], "y")
    assert not [f for f in result["findings"] if f["severity"] == "high"]


def _repo(tmp_path, files):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name, text in files.items():
        (tmp_path / name).write_text(textwrap.dedent(text))
    return build_context(tmp_path)


def test_static_leakage_rules(tmp_path):
    ctx = _repo(tmp_path, {"train.py": """
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import KFold, train_test_split, cross_val_score
        cv = KFold(5)
        parts = "a,b".split(",")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y)
        enc = StandardScaler().fit(X_te)
        df["next_amount"] = df["amount"].shift(-1)
        df = df.bfill()
    """, "ok.py": """
        from sklearn.pipeline import make_pipeline
        X_tr, X_te, y_tr, y_te = train_test_split(X, y)
        scaler = StandardScaler()
        scaler.fit(X_tr)
    """})
    found = {(f.rule, f.path, f.line) for f in leakage.run(ctx).findings}
    rules = {r for r, p, _ in found if p == "train.py"}
    assert rules == {"fit-before-split", "fit-on-holdout", "future-shift", "backfill"}
    assert not [f for f in found if f[1] == "ok.py"]


def test_static_leakage_notebook_cell_order(tmp_path):
    nb = {"cells": [
        {"cell_type": "code", "source": "imp = SimpleImputer()\nX = imp.fit_transform(X)", "outputs": []},
        {"cell_type": "code", "source": "scores = cross_val_score(model, X, y)", "outputs": []},
    ]}
    ctx = _repo(tmp_path, {"nb.ipynb": json.dumps(nb)})
    findings = leakage.run(ctx).findings
    assert [(f.rule, f.cell) for f in findings] == [("fit-before-split", 0)]


# --- llm_eval -------------------------------------------------------------------

FAKE_TARGET = """
import json
CALLS = {"n": 0}
def run(inp):
    CALLS["n"] += 1
    if inp == "boom":
        raise RuntimeError("gateway timeout")
    if isinstance(inp, dict):
        return {"output": json.dumps({"name": inp["name"], "year": 2020}),
                "usage": {"input_tokens": 100, "output_tokens": 20}, "model": "fake-1"}
    return inp.upper() + "."
def judge(case, output):
    return {"score": 1.0 if "A" in str(output) else 0.0, "reason": "has A"}
"""


def _cases(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return str(path)


def test_llm_eval_run_scores_caches_and_diffs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "fake_target.py").write_text(FAKE_TARGET)
    cases = _cases(tmp_path / "golden.jsonl", [
        {"id": "a", "input": "abc", "expected": "ABC", "tags": ["easy"], "human_score": 1},
        {"id": "b", "input": {"name": "Ada"}, "expected": {"name": "Ada", "year": 2021}},
        {"id": "c", "input": "boom", "expected": "x"},
        {"id": "d", "input": "z", "checks": [{"type": "regex", "value": "^Z"}], "human_score": 0},
    ])
    assert llm_eval.main(["run", "--cases", cases, "--target", "fake_target:run", "--judge", "fake_target:judge",
                          "--name", "first", "--price-in", "3", "--price-out", "15"]) == 0
    first = json.loads((tmp_path / ".dsguard/evals/first.json").read_text())
    s = first["summary"]
    assert s["pass_rate"] == pytest.approx(0.5)            # a, d pass; b (year wrong), c (error) fail
    assert s["by_check"]["fields"] == 0 and s["error_rate"] == pytest.approx(0.25)
    assert s["input_tokens"] == 100 and s["est_cost"] == pytest.approx((100 * 3 + 20 * 15) / 1e6)
    b = next(r for r in first["results"] if r["id"] == "b")
    assert b["score"] == pytest.approx(0.5)

    import fake_target
    calls = fake_target.CALLS["n"]
    cases2 = _cases(tmp_path / "golden2.jsonl", [
        {"id": "a", "input": "abc", "expected": "XYZ"},
        {"id": "b", "input": {"name": "Ada"}, "expected": {"name": "Ada", "year": 2020}},
    ])
    code = llm_eval.main(["run", "--cases", cases2, "--target", "fake_target:run", "--name", "second",
                          "--baseline", ".dsguard/evals/first.json"])
    assert code == 1                                         # 'a' regressed
    second = json.loads((tmp_path / ".dsguard/evals/second.json").read_text())
    assert second["baseline"]["regressions"] == ["a"] and second["baseline"]["fixes"] == ["b"]
    assert fake_target.CALLS["n"] == calls                   # both served from cache


def test_llm_eval_checks():
    assert llm_eval.run_check({"type": "numeric", "tol": 0.01}, "about 99.5 units", 100)[0]
    assert llm_eval.run_check({"type": "json_valid"}, '```json\n{"a": 1}\n```', None)[0]
    passed, score, _ = llm_eval.run_check({"type": "contains", "value": ["alpha", "beta"]}, "Alpha only", None)
    assert not passed and score == 0.5
    assert llm_eval.kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
