"""Tests for drift_monitor and the facts ledger behind stakeholder summaries."""

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))

import drift_monitor  # noqa: E402
import facts  # noqa: E402

RNG = np.random.default_rng(11)


def _frame(n, shift=0.0, new_plan=False, start="2025-01-01", days=90, signal=2.0):
    x = RNG.normal(shift, 1, n)
    z = RNG.normal(0, 1, n)
    plans = RNG.choice(["basic", "pro"] + (["enterprise"] if new_plan else []), n)
    p = 1 / (1 + np.exp(-2 * x))
    y = (RNG.uniform(size=n) < 1 / (1 + np.exp(-signal * x))).astype(float)
    ts = pd.Timestamp(start) + pd.to_timedelta(RNG.integers(0, days, n), unit="D")
    return pd.DataFrame({"x": x, "z": z, "plan": plans, "p": p, "y": y, "ts": ts})


def test_stable_data_is_ok():
    ref, cur = _frame(5000), _frame(3000)
    result = drift_monitor.run_monitor(ref, cur, pred="p", target="y", time_col="ts")
    assert result["status"] == "OK", result["reasons"]
    assert all(f["status"] == "stable" for f in result["features"])
    assert result["periods"][-1]["partial"] and not result["periods"][0]["partial"]


def test_volume_drop_in_complete_period():
    ref = _frame(4000)
    cur = _frame(4000, start="2025-01-06", days=56)              # eight full ISO weeks, Mon..Sun
    last_week = pd.to_datetime(cur["ts"]) >= pd.Timestamp("2025-02-24")
    cur = pd.concat([cur[~last_week], cur[last_week].head(40)])
    result = drift_monitor.run_monitor(ref, cur, pred="p", time_col="ts")
    assert not result["periods"][-1]["partial"]
    assert any("latest complete period" in r for r in result["reasons"]) and result["status"] != "OK"


def test_shift_new_category_and_performance_drop_alert():
    ref = _frame(5000)
    cur = _frame(3000, shift=1.0, new_plan=True, signal=0.3)    # x shifts, new plan, labels decouple
    importance = {"x": 0.7, "plan": 0.2, "z": 0.1}
    result = drift_monitor.run_monitor(ref, cur, pred="p", target="y", importance=importance)
    by_feature = {f["feature"]: f for f in result["features"]}
    assert by_feature["x"]["status"] == "major" and by_feature["plan"]["status"] == "major"
    assert by_feature["plan"]["new_category_share"] > 0.2
    assert by_feature["z"]["status"] == "stable"
    assert result["performance"]["status"] == "major"
    assert result["status"] == "ALERT"
    assert any("important features" in r for r in result["reasons"])


def test_schema_and_missingness():
    ref = _frame(3000)
    cur = _frame(2000).drop(columns=["z"])
    cur.loc[cur.sample(frac=0.3, random_state=0).index, "x"] = np.nan
    result = drift_monitor.run_monitor(ref, cur, pred="p")
    assert result["schema"]["missing_columns"] == ["z"] and result["status"] == "ALERT"
    x = next(f for f in result["features"] if f["feature"] == "x")
    assert x["status"] == "major" and x["missing_delta"] == pytest.approx(0.3, abs=0.02)


def test_labels_not_arrived():
    ref, cur = _frame(3000), _frame(3000)
    cur["y"] = np.nan
    result = drift_monitor.run_monitor(ref, cur, pred="p", target="y")
    assert result["performance"]["status"] == "insufficient-labels"
    assert result["performance"]["label_coverage"] == 0


def test_cli_config_history_and_exit_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _frame(4000).to_csv("ref.csv", index=False)
    _frame(4000, shift=1.5, days=60, start="2025-06-01").to_csv("cur.csv", index=False)
    Path("monitor.toml").write_text('reference = "ref.csv"\ncurrent = "cur.csv"\npred = "p"\ntime = "ts"\n'
                                    'exclude = ["y"]\n[thresholds]\npsi_major = 0.3\n')
    code = drift_monitor.main(["--config", "monitor.toml", "--out", "out"])
    report = json.loads(Path("out/drift.json").read_text())
    assert report["thresholds"]["psi_major"] == 0.3
    assert report["status"] == "ALERT" and code == 1
    assert report["periods"] and "psi:p" in report["periods"][0]
    history = Path(".dsguard/drift/history.jsonl").read_text().splitlines()
    assert json.loads(history[-1])["status"] == "ALERT"


# --- facts ----------------------------------------------------------------------

DIAG = {"task": "binary", "rows": 12000, "metrics": {"auc": 0.8412, "ece": 0.0712, "prevalence": 0.183},
        "flags": [{"severity": "medium", "rule": "miscalibrated", "message": "ECE 0.071: probabilities are off by ~7% on average"}],
        "segments": [{"segment": "region", "level": "east", "n": 840, "metric": 0.71}]}


@pytest.fixture
def ledger(tmp_path):
    report = tmp_path / "diagnostics.json"
    report.write_text(json.dumps(DIAG))
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"summary": {"pass_rate": 0.80}}))
    new = tmp_path / "new.json"
    new.write_text(json.dumps({"summary": {"pass_rate": 0.86}}))
    out = tmp_path / "facts.json"
    facts.main(["collect", str(report), "--compare", str(old), str(new), "--out", str(out)])
    data = facts.load_facts(str(out))
    ids = {f["label"]: f["id"] for f in data.values()}
    return out, ids


def test_collect_detects_report_types(ledger):
    out, ids = ledger
    assert {"model auc", "model ece", "evaluation rows", "summary.pass_rate change"} <= set(ids)
    assert any(label.startswith("diagnostic flag") for label in ids)


def test_verify_accepts_cited_numbers_in_any_display_form(ledger, tmp_path):
    out, ids = ledger
    auc, ece, rows = ids["model auc"], ids["model ece"], ids["evaluation rows"]
    change = ids["summary.pass_rate change"]
    summary = tmp_path / "s.md"
    summary.write_text(
        f"# Churn model review 2026-10-20\n\n"
        f"The model ranks churners well (AUC 0.84 [{auc}]) on 12,000 held-out customers [{rows}].\n"
        f"Probabilities are about 7.1% off on average [{ece}], so we recalibrate before pricing.\n"
        f"- Extraction pass rate improved by 6pp [{change}].\n"
        "Next we run the leakage audit.\n"
    )
    assert facts.main(["verify", str(summary), "--facts", str(out)]) == 0
    final = tmp_path / "final.md"
    assert facts.main(["render", str(summary), "--facts", str(out), "--out", str(final)]) == 0
    text = final.read_text()
    assert "[F" not in text and "Sources:" in text and "12,000" in text


def test_verify_rejects_invented_uncited_and_dangling(ledger, tmp_path):
    out, ids = ledger
    auc = ids["model auc"]
    summary = tmp_path / "bad.md"
    summary.write_text(
        f"AUC is 0.91 [{auc}].\n"              # wrong number for the cited fact
        "Revenue at risk is $2.3M.\n"          # no citation
        "Calibration is fine [F999].\n"        # dangling citation (no number, still an error)
    )
    problems = facts.verify(summary.read_text(), facts.load_facts(str(out)))
    assert len(problems) == 3
    assert facts.main(["render", str(summary), "--facts", str(out), "--out", str(tmp_path / "x.md")]) == 1


def test_matches_rounding_and_units():
    assert facts.matches(7.0, 0, "%", [0.0712])
    assert facts.matches(0.07, 2, None, [0.0712])
    assert not facts.matches(0.08, 2, None, [0.0712])
    assert facts.matches(2.3, 1, "M", [2_310_000])
    assert facts.matches(0.02, 2, None, [-0.02])
