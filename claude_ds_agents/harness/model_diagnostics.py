#!/usr/bin/env python3
"""Deterministic model diagnostics from an evaluation table.

Input is framework-agnostic: one row per scored example with the target, the
prediction (probability for binary, value for regression), the features, and
optionally segment, time and id columns. Any model (sklearn, LightGBM, PyTorch,
an LLM classifier) can be diagnosed once its predictions are in that table.

  python model_diagnostics.py --data eval.parquet --target churned --pred p_churn \
      --segments region plan --time scored_at --id customer_id [--model model.joblib]

Writes diagnostics.json, diagnostics.md and (with matplotlib) PNG plots to
.dsguard/diagnostics/<timestamp>/. The model-diagnostics agent interprets them.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    TIME_UNITS, binary_metrics, calibration_table, is_binary_target, load_table, md_table, out_dir,
    regression_metrics, score, single_feature_strength, time_frequency, write_report,
)

MIN_SEGMENT_N = 30


def residuals(task: str, y: pd.Series, pred: pd.Series) -> pd.Series:
    """y - prediction. For binary this is the probability residual (observed - predicted)."""
    return y.astype(float) - pred.astype(float)


def feature_profile(frame: pd.DataFrame, features: list[str], y: pd.Series, binary: bool) -> pd.DataFrame:
    rows = []
    for col in features:
        x = frame[col]
        rows.append({
            "feature": col,
            "dtype": str(x.dtype),
            "missing_rate": float(x.isna().mean()),
            "n_unique": int(x.nunique(dropna=True)),
            "constant": bool(x.nunique(dropna=True) <= 1),
            "single_feature_strength": single_feature_strength(x, y, binary),
        })
    return pd.DataFrame(rows).sort_values("single_feature_strength", ascending=False, na_position="last")


def correlated_pairs(frame: pd.DataFrame, features: list[str], threshold: float = 0.95) -> pd.DataFrame:
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(frame[c]) and frame[c].nunique() > 1]
    if len(numeric) < 2:
        return pd.DataFrame(columns=["a", "b", "spearman"])
    corr = frame[numeric].rank().corr()
    pairs = []
    for i, a in enumerate(numeric):
        for b in numeric[i + 1:]:
            value = corr.loc[a, b]
            if abs(value) >= threshold:
                pairs.append({"a": a, "b": b, "spearman": float(value)})
    return pd.DataFrame(pairs, columns=["a", "b", "spearman"])


def residual_structure(frame: pd.DataFrame, features: list[str], resid: pd.Series, bins: int = 10) -> tuple[pd.DataFrame, dict]:
    """Per feature: share of residual variance explained by binning on that feature (eta squared).

    A model that captured a feature fully leaves mean residual ~0 in every bin; a high
    eta squared means systematic error along that feature (missed non-linearity,
    interaction, or a segment the model under-serves).
    """
    total_var = float(resid.var())
    summary, detail = [], {}
    for col in features:
        x = frame[col]
        if pd.api.types.is_numeric_dtype(x) and x.nunique() > bins:
            groups = pd.qcut(x, bins, duplicates="drop").astype(str)
        else:
            top = x.astype(str).value_counts().index[:20]
            groups = x.astype(str).where(x.astype(str).isin(top), "<other>")
        groups = groups.where(x.notna(), "<missing>")
        table = resid.groupby(groups).agg(["size", "mean", "std"]).rename(columns={"size": "n"})
        between = float((table["n"] * table["mean"] ** 2).sum() / table["n"].sum() - resid.mean() ** 2)
        eta2 = between / total_var if total_var > 0 else float("nan")
        worst = table["mean"].abs().idxmax() if not table.empty else None
        summary.append({
            "feature": col, "eta_squared": eta2, "worst_bin": worst,
            "worst_bin_mean_residual": float(table.loc[worst, "mean"]) if worst is not None else float("nan"),
            "worst_bin_n": int(table.loc[worst, "n"]) if worst is not None else 0,
        })
        detail[col] = table.reset_index(names="bin").to_dict(orient="records")
    result = pd.DataFrame(summary).sort_values("eta_squared", ascending=False)
    return result, detail


def segment_metrics(frame: pd.DataFrame, task: str, target: str, pred: str, segments: list[str]) -> pd.DataFrame:
    rows = []
    overall = score(task, frame[target], frame[pred])
    for col in segments:
        for level, part in frame.groupby(frame[col].astype(str), observed=True):
            if len(part) < MIN_SEGMENT_N or (task == "binary" and part[target].nunique() < 2):
                continue
            value = score(task, part[target], part[pred])
            rows.append({
                "segment": col, "level": level, "n": len(part),
                "metric": value if task == "binary" else -value,
                "gap_vs_overall": value - overall,
                "mean_residual": float((part[target].astype(float) - part[pred].astype(float)).mean()),
            })
    if not rows:
        return pd.DataFrame(columns=["segment", "level", "n", "metric", "gap_vs_overall", "mean_residual"])
    return pd.DataFrame(rows).sort_values("gap_vs_overall")


def time_metrics(frame: pd.DataFrame, task: str, target: str, pred: str, time_col: str) -> tuple[pd.DataFrame, float]:
    times = pd.to_datetime(frame[time_col], errors="coerce")
    freq = time_frequency(times.dropna())
    rows = []
    for period, part in frame.assign(_t=times).dropna(subset=["_t"]).groupby(pd.Grouper(key="_t", freq=freq)):
        if len(part) < MIN_SEGMENT_N or (task == "binary" and part[target].nunique() < 2):
            continue
        value = score(task, part[target], part[pred])
        rows.append({"period": period.date().isoformat(), "n": len(part),
                     "metric": value if task == "binary" else -value,
                     "mean_residual": float((part[target].astype(float) - part[pred].astype(float)).mean())})
    table = pd.DataFrame(rows, columns=["period", "n", "metric", "mean_residual"])
    slope = float(np.polyfit(np.arange(len(table)), table["metric"], 1)[0]) if len(table) >= 3 else float("nan")
    return table, slope


def permutation_importance(model, frame: pd.DataFrame, features: list[str], task: str, target: str,
                           repeats: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def predict(data):
        if task == "binary" and hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(data))[:, 1]
        return np.asarray(model.predict(data), dtype=float)

    base_x = frame[features]
    baseline = score(task, frame[target], predict(base_x))
    rows = []
    for col in features:
        drops = []
        for _ in range(repeats):
            shuffled = base_x.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            drops.append(baseline - score(task, frame[target], predict(shuffled)))
        rows.append({"feature": col, "importance": float(np.mean(drops)), "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def plots(directory: Path, task: str, y: pd.Series, pred: pd.Series, resid: pd.Series, calib) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    made = []
    if task == "binary":
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
        ax.plot(calib["mean_pred"], calib["observed"], marker="o")
        ax.set(xlabel="mean predicted", ylabel="observed rate", title="Calibration")
        fig.savefig(directory / "calibration.png", dpi=120, bbox_inches="tight")
        made.append("calibration.png")
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(pred, resid, s=4, alpha=0.4)
        ax.axhline(0, color="grey", linestyle="--")
        ax.set(xlabel="prediction", ylabel="residual", title="Residuals vs prediction")
        fig.savefig(directory / "residuals_vs_pred.png", dpi=120, bbox_inches="tight")
        made.append("residuals_vs_pred.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(resid, bins=50)
    ax.set(xlabel="residual", title="Residual distribution")
    fig.savefig(directory / "residual_hist.png", dpi=120, bbox_inches="tight")
    made.append("residual_hist.png")
    plt.close("all")
    return made


def flags(task, metrics, profile, structure, segments, slope, pairs) -> list[dict]:
    """Rule-based observations; thresholds are conventions the agent may challenge in context."""
    out = []

    def add(severity, rule, message):
        out.append({"severity": severity, "rule": rule, "message": message})

    if task == "binary":
        if metrics["ece"] > 0.05:
            add("medium", "miscalibrated", f"ECE {metrics['ece']:.3f}: probabilities are off by ~{metrics['ece']:.0%} on average")
        if abs(metrics["mean_prediction"] - metrics["prevalence"]) > 0.02:
            add("medium", "base-rate-mismatch",
                f"mean prediction {metrics['mean_prediction']:.3f} vs prevalence {metrics['prevalence']:.3f}")
        if metrics["auc"] >= 0.98:
            add("high", "too-good", f"AUC {metrics['auc']:.3f} is rarely real on tabular data; check for leakage")
    else:
        if abs(metrics["bias_mean_residual"]) > 0.1 * metrics["mae"]:
            add("medium", "biased", f"mean residual {metrics['bias_mean_residual']:.4g} (systematic over/under-prediction)")
        if abs(metrics["residual_skew"]) > 1:
            add("low", "skewed-residuals", f"residual skew {metrics['residual_skew']:.2f}; consider a transformed target or a different loss")
        if not math.isnan(metrics["r2"]) and metrics["r2"] >= 0.99:
            add("high", "too-good", f"R² {metrics['r2']:.3f}; check for leakage")
    for row in profile.itertuples():
        if not math.isnan(row.single_feature_strength) and row.single_feature_strength >= 0.8:
            add("high", "dominant-feature",
                f"{row.feature} alone has strength {row.single_feature_strength:.2f}; verify it is known at prediction time")
        if row.constant:
            add("low", "constant-feature", f"{row.feature} is constant in this data")
        if row.missing_rate > 0.5:
            add("low", "mostly-missing", f"{row.feature} is {row.missing_rate:.0%} missing")
    for row in structure.head(5).itertuples():
        if row.eta_squared >= 0.01:
            add("medium", "residual-structure",
                f"residuals vary systematically with {row.feature} (eta² {row.eta_squared:.3f}; worst bin {row.worst_bin} "
                f"mean residual {row.worst_bin_mean_residual:.3g}, n={row.worst_bin_n})")
    # Segment gaps: AUC points for binary, RMSE difference (as a share of overall RMSE) for regression.
    min_gap = 0.05 if task == "binary" else 0.2 * metrics["rmse"]
    for row in segments.head(5).itertuples():
        if row.gap_vs_overall <= -min_gap:
            add("medium", "weak-segment", f"{row.segment}={row.level} (n={row.n}) metric {row.metric:.3f}, gap {row.gap_vs_overall:+.3f}")
    if not math.isnan(slope):
        trend = slope if task == "binary" else -slope
        if trend < 0 and abs(slope) > (0.005 if task == "binary" else 0.01 * metrics["rmse"]):
            add("medium", "degrading-over-time", f"metric trend {slope:+.4f} per period")
    for row in pairs.itertuples():
        add("low", "redundant-features", f"{row.a} and {row.b} are near-duplicates (spearman {row.spearman:.3f})")
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(out, key=lambda f: order[f["severity"]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--pred", required=True, help="prediction column (probability of class 1 for binary)")
    ap.add_argument("--task", choices=["auto", "binary", "regression"], default="auto")
    ap.add_argument("--features", nargs="*", help="default: every other column")
    ap.add_argument("--segments", nargs="*", default=[])
    ap.add_argument("--time")
    ap.add_argument("--id")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--model", help="optional pickled/joblib model with predict[_proba] for permutation importance")
    ap.add_argument("--sample", type=int, default=200_000, help="row cap for speed")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    frame = load_table(args.data)
    if len(frame) > args.sample:
        frame = frame.sample(args.sample, random_state=0)
    frame = frame.dropna(subset=[args.target, args.pred]).reset_index(drop=True)
    task = args.task if args.task != "auto" else ("binary" if is_binary_target(frame[args.target]) else "regression")
    y, pred = frame[args.target].astype(float), frame[args.pred].astype(float)
    reserved = {args.target, args.pred, args.time, args.id, *args.exclude}
    features = args.features or [c for c in frame.columns if c not in reserved]

    resid = residuals(task, y, pred)
    if task == "binary":
        metrics = binary_metrics(y, pred)
        calib, _ = calibration_table(y, pred)
        thresholds = pd.DataFrame([
            {"threshold": t, "precision": float(y[pred >= t].mean()) if (pred >= t).any() else float("nan"),
             "recall": float(((pred >= t) & (y == 1)).sum() / max(y.sum(), 1)),
             "flagged_share": float((pred >= t).mean())}
            for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        ])
        worst_score = -(y * np.log(np.clip(pred, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - pred, 1e-15, 1)))
    else:
        metrics = regression_metrics(y, pred)
        calib, thresholds = pd.DataFrame(), pd.DataFrame()
        metrics["abs_residual_vs_pred_spearman"] = float(resid.abs().rank().corr(pred.rank()))
        worst_score = resid.abs()

    profile = feature_profile(frame, features, y, task == "binary")
    structure, structure_detail = residual_structure(frame, features, resid)
    segments = segment_metrics(frame, task, args.target, args.pred, args.segments)
    time_table, slope = time_metrics(frame, task, args.target, args.pred, args.time) if args.time else (pd.DataFrame(), float("nan"))
    pairs = correlated_pairs(frame, features)
    worst_cols = [c for c in (args.id, args.target, args.pred) if c]
    worst = frame.assign(_err=worst_score).nlargest(20, "_err")[worst_cols + ["_err"]]

    importance = pd.DataFrame()
    notes = []
    if args.model:
        try:
            import joblib
            model = joblib.load(args.model)
        except ImportError:
            import pickle
            with open(args.model, "rb") as fh:
                model = pickle.load(fh)
        importance = permutation_importance(model, frame, features, task, args.target)
    else:
        notes.append("no --model given: permutation importance skipped")

    directory = out_dir(args.out, "diagnostics")
    images = plots(directory, task, y, pred, resid, calib)
    if not images:
        notes.append("matplotlib not installed: plots skipped")
    found = flags(task, metrics, profile, structure, segments, slope, pairs)

    payload = {
        "task": task, "data": args.data, "rows": len(frame), "metrics": metrics, "flags": found,
        "feature_profile": profile.to_dict(orient="records"),
        "residual_structure": structure.to_dict(orient="records"), "residual_bins": structure_detail,
        "segments": segments.to_dict(orient="records"), "time": time_table.to_dict(orient="records"),
        "time_slope": slope, "calibration": calib.to_dict(orient="records"),
        "thresholds": thresholds.to_dict(orient="records"), "correlated_pairs": pairs.to_dict(orient="records"),
        "permutation_importance": importance.to_dict(orient="records"),
        "worst_predictions": worst.to_dict(orient="records"), "plots": images, "notes": notes,
    }
    md = [f"# Model diagnostics ({task})\n", f"Data: `{args.data}`, rows: {len(frame)}\n", "## Metrics\n",
          md_table(pd.DataFrame([metrics])), "## Flags\n"]
    md += [f"- **{f['severity'].upper()}** [{f['rule']}] {f['message']}\n" for f in found] or ["_none_\n"]
    md += ["\n## Residual structure by feature (top 10)\n", md_table(structure, 10)]
    md += ["\n## Feature profile (strongest single features first)\n", md_table(profile, 15)]
    if not segments.empty:
        md += ["\n## Weakest segments\n", md_table(segments, 10)]
    if not time_table.empty:
        md += [f"\n## Over time (per {TIME_UNITS[time_frequency(pd.to_datetime(frame[args.time], errors='coerce').dropna())]}, "
               f"slope {slope:+.4g})\n", md_table(time_table, 30)]
    if task == "binary":
        md += ["\n## Calibration\n", md_table(calib), "\n## Thresholds\n", md_table(thresholds)]
    if not importance.empty:
        md += ["\n## Permutation importance\n", md_table(importance, 20)]
    md += ["\n## Worst predictions\n", md_table(worst, 20)]
    if notes:
        md += ["\n## Notes\n"] + [f"- {n}\n" for n in notes]
    json_path, md_path = write_report(directory, "diagnostics", payload, "".join(md))
    print(f"diagnostics: {md_path}\njson: {json_path}")
    for f in found[:15]:
        print(f"  {f['severity'].upper():6} [{f['rule']}] {f['message']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
