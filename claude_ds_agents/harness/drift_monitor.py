#!/usr/bin/env python3
"""Deterministic drift monitoring: reference (training/baseline) vs current (production) data.

  python drift_monitor.py --reference train.parquet --current scores_last_30d.parquet \
      --pred score [--target label] [--time scored_at --freq W] \
      [--importance .dsguard/diagnostics/<run>/diagnostics.json]
  python drift_monitor.py --config monitor.toml      # same keys as the flags, for cron

Per feature: PSI, KS (numeric), standardized mean shift, out-of-reference-range share,
unseen categories, missing-rate change. Also prediction drift, performance drift where
labels have arrived (with label coverage), schema changes, and per-period trends and
volume when --time is given. Statuses are threshold rules, weighted by feature importance
when provided.

Exit code: 0 OK/WATCH, 1 ALERT (or WATCH with --fail-on watch), so a scheduler can decide
when to wake the drift-monitor agent. Appends one line per run to a history file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import is_binary_target, load_table, md_table, out_dir, psi, score, to_jsonable, write_report  # noqa: E402

DEFAULT_THRESHOLDS = {
    "psi_moderate": 0.1, "psi_major": 0.25,
    "new_category_moderate": 0.01, "new_category_major": 0.05,
    "missing_moderate": 0.02, "missing_major": 0.10,
    "out_of_range_moderate": 0.01, "out_of_range_major": 0.05,
    "auc_drop_moderate": 0.02, "auc_drop_major": 0.05,
    "rmse_rise_moderate": 0.05, "rmse_rise_major": 0.10,   # relative
    "volume_change": 0.5,                                  # vs median of earlier periods
    "important_top_k": 5,
}
RANK = {"stable": 0, "moderate": 1, "major": 2}
PERIOD_ALIASES = {"MS": "M", "M": "M", "W": "W", "D": "D"}


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(a[~np.isnan(a)]), np.sort(b[~np.isnan(b)])
    if not len(a) or not len(b):
        return float("nan")
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, "right") / len(a) - np.searchsorted(b, grid, "right") / len(b))))


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def feature_drift(ref: pd.Series, cur: pd.Series, th: dict) -> dict:
    out = {"psi": psi(ref, cur), "missing_ref": float(ref.isna().mean()), "missing_cur": float(cur.isna().mean())}
    out["missing_delta"] = out["missing_cur"] - out["missing_ref"]
    reasons = []
    level = "stable"

    def bump(new_level, reason):
        nonlocal level
        reasons.append(reason)
        level = max(level, new_level, key=RANK.get)

    if is_numeric(ref) and is_numeric(cur):
        r, c = ref.dropna().to_numpy(float), cur.dropna().to_numpy(float)
        std = float(np.std(r)) if len(r) else float("nan")
        out.update(
            ks=ks_statistic(r, c), mean_ref=float(np.mean(r)) if len(r) else float("nan"),
            mean_cur=float(np.mean(c)) if len(c) else float("nan"),
            out_of_range=float(((c < r.min()) | (c > r.max())).mean()) if len(r) and len(c) else 0.0,
        )
        out["mean_shift_sd"] = (out["mean_cur"] - out["mean_ref"]) / std if std and std > 0 else float("nan")
        oor = out["out_of_range"]
        if oor >= th["out_of_range_major"]:
            bump("major", f"{oor:.1%} of values outside the reference range")
        elif oor >= th["out_of_range_moderate"]:
            bump("moderate", f"{oor:.1%} of values outside the reference range")
    else:
        seen = set(ref.dropna().astype(str))
        cur_values = cur.dropna().astype(str)
        unseen = cur_values[~cur_values.isin(seen)]
        out["new_category_share"] = float(len(unseen) / len(cur_values)) if len(cur_values) else 0.0
        out["new_categories"] = unseen.value_counts().head(5).index.tolist()
        share = out["new_category_share"]
        if share >= th["new_category_major"]:
            bump("major", f"{share:.1%} of values are categories unseen in reference")
        elif share >= th["new_category_moderate"]:
            bump("moderate", f"{share:.1%} of values are categories unseen in reference")
    value = out["psi"]
    if not math.isnan(value):
        if value >= th["psi_major"]:
            bump("major", f"PSI {value:.2f}")
        elif value >= th["psi_moderate"]:
            bump("moderate", f"PSI {value:.2f}")
    delta = out["missing_delta"]
    if abs(delta) >= th["missing_major"]:
        bump("major", f"missing rate {out['missing_ref']:.1%} -> {out['missing_cur']:.1%}")
    elif abs(delta) >= th["missing_moderate"]:
        bump("moderate", f"missing rate {out['missing_ref']:.1%} -> {out['missing_cur']:.1%}")
    out["status"], out["reasons"] = level, reasons
    return out


def load_importance(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "permutation_importance" in data:   # model_diagnostics output
        data = {r["feature"]: r["importance"] for r in data["permutation_importance"]}
    total = sum(max(v, 0) for v in data.values()) or 1.0
    return {k: max(float(v), 0) / total for k, v in data.items()}


def performance(ref: pd.DataFrame, cur: pd.DataFrame, target: str, pred: str, th: dict) -> dict:
    labelled = cur.dropna(subset=[target, pred])
    out = {"label_coverage": float(cur[target].notna().mean()), "labelled_rows": len(labelled)}
    ref_l = ref.dropna(subset=[target, pred])
    if len(labelled) < 30 or len(ref_l) < 30:
        out["status"] = "insufficient-labels"
        return out
    task = "binary" if is_binary_target(ref_l[target]) else "regression"
    if task == "binary" and labelled[target].nunique() < 2:
        out["status"] = "insufficient-labels"
        return out
    r, c = score(task, ref_l[target], ref_l[pred]), score(task, labelled[target], labelled[pred])
    out["task"] = task
    if task == "binary":
        out.update(metric="auc", reference=r, current=c, change=c - r)
        drop = r - c
        out["status"] = "major" if drop >= th["auc_drop_major"] else "moderate" if drop >= th["auc_drop_moderate"] else "stable"
    else:
        rise = (-c - -r) / -r if r else 0.0
        out.update(metric="rmse", reference=-r, current=-c, change=-c - -r, relative_change=rise)
        out["status"] = "major" if rise >= th["rmse_rise_major"] else "moderate" if rise >= th["rmse_rise_moderate"] else "stable"
    return out


def per_period(ref: pd.DataFrame, cur: pd.DataFrame, time_col: str, freq: str, columns: list[str],
               target: str | None, pred: str | None, th: dict) -> tuple[pd.DataFrame, list[str]]:
    times = pd.to_datetime(cur[time_col], errors="coerce")
    periods = times.dt.to_period(PERIOD_ALIASES.get(freq, freq))
    last_day = times.max().normalize()
    rows = []
    for period, part in cur.assign(_p=periods).dropna(subset=["_p"]).groupby("_p"):
        # A period still in progress has fewer rows by construction; flag it so volume checks skip it.
        row = {"period": str(period), "rows": len(part), "partial": bool(period.end_time.normalize() > last_day)}
        for col in columns:
            row[f"psi:{col}"] = psi(ref[col], part[col])
        if target and pred:
            labelled = part.dropna(subset=[target, pred])
            ok = len(labelled) >= 30 and (not is_binary_target(ref[target].dropna()) or labelled[target].nunique() == 2)
            task = "binary" if is_binary_target(ref[target].dropna()) else "regression"
            row["metric"] = (score(task, labelled[target], labelled[pred]) * (1 if task == "binary" else -1)) if ok else float("nan")
            row["label_coverage"] = float(part[target].notna().mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    alerts = []
    complete = table[~table["partial"]] if not table.empty else table
    if len(complete) >= 4:
        latest, history = complete["rows"].iloc[-1], complete["rows"].iloc[:-1].median()
        if history and abs(latest - history) / history >= th["volume_change"]:
            alerts.append(f"latest complete period ({complete['period'].iloc[-1]}) has {latest} rows vs median {history:.0f}")
    return table, alerts


def overall_status(features: dict, importance: dict, score_drift: dict | None, perf: dict | None,
                   missing_cols: list[str], th: dict) -> tuple[str, list[str]]:
    reasons = []
    if missing_cols:
        reasons.append(f"columns missing from current data: {', '.join(missing_cols)}")
    if score_drift and score_drift["status"] == "major":
        reasons.append(f"prediction distribution shifted (PSI {score_drift['psi']:.2f})")
    if perf and perf.get("status") == "major":
        reasons.append(f"{perf['metric']} {perf['reference']:.3f} -> {perf['current']:.3f}")
    major = [f for f, d in features.items() if d["status"] == "major"]
    if importance:
        top = sorted(importance, key=importance.get, reverse=True)[: th["important_top_k"]]
        important_major = [f for f in major if f in top]
        if important_major:
            reasons.append(f"major drift in important features: {', '.join(important_major)}")
    elif features and len(major) / len(features) >= 0.25:
        reasons.append(f"{len(major)} of {len(features)} features drifted majorly")
    if reasons:
        return "ALERT", reasons
    watch = []
    if major:
        watch.append(f"major drift in: {', '.join(major[:8])}")
    if score_drift and score_drift["status"] == "moderate":
        watch.append(f"prediction drift PSI {score_drift['psi']:.2f}")
    if perf and perf.get("status") == "moderate":
        watch.append(f"{perf['metric']} {perf['reference']:.3f} -> {perf['current']:.3f}")
    moderate = [f for f, d in features.items() if d["status"] == "moderate"]
    if moderate:
        watch.append(f"moderate drift in: {', '.join(moderate[:8])}")
    return ("WATCH", watch) if watch else ("OK", [])


def run_monitor(ref: pd.DataFrame, cur: pd.DataFrame, pred=None, target=None, time_col=None, freq="W",
                features=None, exclude=(), importance=None, thresholds=None) -> dict:
    th = DEFAULT_THRESHOLDS | (thresholds or {})
    importance = importance or {}
    reserved = {pred, target, time_col, *exclude}
    feats = features or [c for c in ref.columns if c not in reserved]
    missing_cols = [c for c in feats if c not in cur.columns]
    new_cols = [c for c in cur.columns if c not in ref.columns and c not in reserved]
    dtype_changes = [{"feature": c, "reference": str(ref[c].dtype), "current": str(cur[c].dtype)}
                     for c in feats if c in cur.columns and is_numeric(ref[c]) != is_numeric(cur[c])]
    present = [c for c in feats if c in cur.columns]
    drift = {c: feature_drift(ref[c], cur[c], th) | {"importance": importance.get(c)} for c in present}
    score_drift = feature_drift(ref[pred], cur[pred], th) if pred and pred in ref and pred in cur else None
    perf = performance(ref, cur, target, pred, th) if target and pred and target in cur and target in ref else None

    status, reasons = overall_status(drift, importance, score_drift, perf, missing_cols, th)
    period_table, volume_alerts = pd.DataFrame(), []
    if time_col and time_col in cur:
        ranked = sorted(present, key=lambda c: (RANK[drift[c]["status"]], drift[c]["psi"] if not math.isnan(drift[c]["psi"]) else 0), reverse=True)
        cols = ([pred] if score_drift else []) + ranked[:3]
        period_table, volume_alerts = per_period(ref, cur, time_col, freq, cols, target, pred, th)
        if volume_alerts and status == "OK":
            status = "WATCH"
        reasons += volume_alerts
    if dtype_changes and status == "OK":
        status = "WATCH"
        reasons.append(f"type changed for: {', '.join(d['feature'] for d in dtype_changes)}")

    table = pd.DataFrame([
        {"feature": c, "status": d["status"], "psi": d["psi"], "importance": d["importance"],
         "ks": d.get("ks"), "mean_shift_sd": d.get("mean_shift_sd"), "missing_delta": d["missing_delta"],
         "new_category_share": d.get("new_category_share"), "reasons": "; ".join(d["reasons"])}
        for c, d in drift.items()
    ])
    if not table.empty:
        table["_rank"] = table["status"].map(RANK)
        table = table.sort_values(["_rank", "psi"], ascending=False).drop(columns="_rank")
    return {
        "status": status, "reasons": reasons, "rows_reference": len(ref), "rows_current": len(cur),
        "schema": {"missing_columns": missing_cols, "new_columns": new_cols, "type_changes": dtype_changes},
        "prediction_drift": score_drift, "performance": perf,
        "features": table.to_dict(orient="records"), "periods": period_table.to_dict(orient="records"),
        "thresholds": th,
    }


def markdown(result: dict, args) -> str:
    md = [f"# Drift monitor: {result['status']}\n",
          f"reference `{args.reference}` ({result['rows_reference']} rows) vs current `{args.current}` "
          f"({result['rows_current']} rows)\n\n"]
    md += [f"- {r}\n" for r in result["reasons"]] or ["- no drift above thresholds\n"]
    schema = result["schema"]
    if any(schema.values()):
        md += ["\n## Schema\n", f"- missing: {schema['missing_columns'] or 'none'}\n",
               f"- new: {schema['new_columns'] or 'none'}\n", f"- type changes: {schema['type_changes'] or 'none'}\n"]
    if result["performance"]:
        md += ["\n## Performance\n", md_table(pd.DataFrame([result["performance"]]))]
    if result["prediction_drift"]:
        pdrift = {k: v for k, v in result["prediction_drift"].items() if k != "reasons"}
        md += ["\n## Prediction drift\n", md_table(pd.DataFrame([pdrift]))]
    md += ["\n## Features (most drifted first)\n", md_table(pd.DataFrame(result["features"]), 25)]
    if result["periods"]:
        md += ["\n## By period\n", md_table(pd.DataFrame(result["periods"]), 30)]
    return "".join(md)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="TOML with the same keys as these flags (dashes as underscores)")
    ap.add_argument("--reference")
    ap.add_argument("--current")
    ap.add_argument("--pred")
    ap.add_argument("--target")
    ap.add_argument("--time")
    ap.add_argument("--freq", default=None, help="period for trends: D, W or MS (default W)")
    ap.add_argument("--features", nargs="*")
    ap.add_argument("--exclude", nargs="*")
    ap.add_argument("--importance", help="{feature: weight} JSON or a model_diagnostics diagnostics.json")
    ap.add_argument("--history", help="JSONL appended per run (default .dsguard/drift/history.jsonl)")
    ap.add_argument("--fail-on", choices=["alert", "watch"], default=None)
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    thresholds = {}
    if args.config:
        config = tomllib.loads(Path(args.config).read_text())
        thresholds = config.pop("thresholds", {})
        for key, value in config.items():
            if getattr(args, key, None) in (None, []):
                setattr(args, key, value)
    if not args.reference or not args.current:
        ap.error("--reference and --current are required (flags or config)")
    args.freq = args.freq or "W"
    args.fail_on = args.fail_on or "alert"

    ref, cur = load_table(args.reference), load_table(args.current)
    result = run_monitor(ref, cur, args.pred, args.target, args.time, args.freq, args.features,
                         args.exclude or [], load_importance(args.importance), thresholds)
    directory = out_dir(args.out, "drift")
    json_path, md_path = write_report(directory, "drift", result, markdown(result, args))

    history = Path(args.history or ".dsguard/drift/history.jsonl")
    history.parent.mkdir(parents=True, exist_ok=True)
    perf = result["performance"] or {}
    entry = {"run_at": datetime.now().isoformat(timespec="seconds"), "status": result["status"],
             "reference": args.reference, "current": args.current, "rows_current": result["rows_current"],
             "prediction_psi": (result["prediction_drift"] or {}).get("psi"),
             "metric": perf.get("metric"), "metric_current": perf.get("current"),
             "major_features": [f["feature"] for f in result["features"] if f["status"] == "major"],
             "report": str(json_path)}
    with history.open("a") as fh:
        fh.write(json.dumps(to_jsonable(entry)) + "\n")

    print(f"drift: {result['status']}  report {md_path}")
    for reason in result["reasons"]:
        print(f"  - {reason}")
    failing = {"alert": ("ALERT",), "watch": ("ALERT", "WATCH")}[args.fail_on]
    return 1 if result["status"] in failing else 0


if __name__ == "__main__":
    sys.exit(main())
