#!/usr/bin/env python3
"""Deterministic train/test leakage checks on the actual data.

  python leakage_check.py --train train.parquet --test test.parquet --target y \
      [--id customer_id] [--group household_id] [--time event_ts] [--text note_text]
  python leakage_check.py --data all.parquet --split split_col --train-value train --target y ...

Checks: shared ids/groups across splits, duplicate feature rows and (near-)duplicate
texts across splits, temporal overlap, feature timestamps after the event time,
columns that predict the target suspiciously well on their own (raw value and
missingness), the target hiding in a feature, row-order/id leakage, and drift.
Writes leakage.json and leakage.md to .dsguard/leakage/<timestamp>/.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import auc, is_binary_target, load_table, md_table, out_dir, psi, single_feature_strength, write_report  # noqa: E402

SUSPECT_NAMES = re.compile(
    r"(?i)(label|target|outcome|result|response|_after|post_|future|next_|final|resolved|closed|"
    r"cancel|churn|default|fraud|status|approved|decision|paid|refund|days_to|time_to)"
)


def row_hashes(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    return pd.util.hash_pandas_object(frame[cols].astype(str), index=False)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(text).lower())).strip()


def bag_key(text: str) -> str:
    return hashlib.sha1(" ".join(sorted(set(normalize_text(text).split()))).encode()).hexdigest()


def split_frames(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.data:
        frame = load_table(args.data)
        is_train = frame[args.split].astype(str) == str(args.train_value)
        return frame[is_train].drop(columns=[args.split]), frame[~is_train].drop(columns=[args.split])
    return load_table(args.train), load_table(args.test)


def run_checks(train: pd.DataFrame, test: pd.DataFrame, target: str, id_col=None, group_col=None,
               time_col=None, text_cols=(), exclude=(), strength_threshold=0.8) -> dict:
    findings: list[dict] = []

    def add(severity, rule, message, **extra):
        findings.append({"severity": severity, "rule": rule, "message": message, **extra})

    reserved = {target, id_col, group_col, time_col, *exclude}
    features = [c for c in train.columns if c not in reserved and c in test.columns]
    binary = is_binary_target(train[target])
    y = train[target]

    # 1. Entity overlap across splits.
    for col, label in ((id_col, "id"), (group_col, "group")):
        if col:
            shared = set(train[col].dropna()) & set(test[col].dropna())
            share = test[col].isin(shared).mean()
            if shared:
                add("high", f"shared-{label}", f"{len(shared)} {label}s ({share:.1%} of test rows) appear in both splits",
                    examples=[str(v) for v in list(shared)[:5]])

    # 2. Duplicate rows across splits (features only, then features + target).
    if features:
        train_hashes = set(row_hashes(train, features))
        dup = row_hashes(test, features).isin(train_hashes)
        if dup.mean() > 0.001:
            add("high" if dup.mean() > 0.01 else "medium", "duplicate-rows",
                f"{dup.mean():.1%} of test rows have an identical feature vector in train")

    # 3. Text duplicates across splits.
    for col in text_cols:
        train_norm = set(train[col].dropna().map(normalize_text))
        test_norm = test[col].dropna().map(normalize_text)
        exact = test_norm.isin(train_norm).mean()
        train_bags = set(train[col].dropna().map(bag_key))
        bag = test[col].dropna().map(bag_key).isin(train_bags).mean()
        if exact > 0.001 or bag > 0.005:
            add("high" if max(exact, bag) > 0.01 else "medium", "duplicate-text",
                f"{col}: {exact:.1%} of test texts match a train text after normalization, "
                f"{bag:.1%} share the same word set (near-duplicates)")

    # 4. Time: overlap and features from the future.
    if time_col:
        t_train = pd.to_datetime(train[time_col], errors="coerce")
        t_test = pd.to_datetime(test[time_col], errors="coerce")
        if t_train.notna().any() and t_test.notna().any():
            before = (t_test < t_train.max()).mean()
            if before > 0:
                add("medium", "temporal-overlap",
                    f"{before:.1%} of test rows are earlier than the latest train row ({t_train.max()}); "
                    "a random split on time-dependent data overstates performance")
        for col in features:
            if pd.api.types.is_datetime64_any_dtype(train[col]) or re.search(r"(?i)(date|_at$|_ts$|time)", col):
                values = pd.to_datetime(train[col], errors="coerce")
                if values.notna().mean() < 0.5:
                    continue
                later = (values > t_train).mean()
                if later > 0.001:
                    add("high", "future-timestamp", f"{col} is after {time_col} in {later:.1%} of train rows: "
                        "information from after the prediction time")

    # 5. Single-feature strength (value and missingness), the target hiding in a feature.
    strengths = []
    y_float = y.astype(float) if binary or pd.api.types.is_numeric_dtype(y) else None
    for col in features:
        x = train[col]
        if x.nunique(dropna=True) <= 1:
            continue
        strength = single_feature_strength(x, y, binary) if y_float is not None else float("nan")
        miss = float("nan")
        if binary and 0.01 < x.isna().mean() < 0.99:
            miss = abs(auc(y_float.to_numpy(), x.isna().to_numpy(dtype=float)) - 0.5) * 2
        strengths.append({"feature": col, "strength": strength, "missingness_strength": miss,
                          "suspect_name": bool(SUSPECT_NAMES.search(col))})
        if y_float is not None and pd.api.types.is_numeric_dtype(x):
            same = (x.astype(float) == y_float).mean()
            inverse = (x.astype(float) == 1 - y_float).mean() if binary else 0
            if max(same, inverse) > 0.99:
                add("high", "target-copy", f"{col} equals the target{' (or its inverse)' if inverse > same else ''} in "
                    f"{max(same, inverse):.1%} of rows")
                continue
        if not math.isnan(strength) and strength >= strength_threshold:
            measure = f"strength {strength:.2f}, AUC {0.5 + strength / 2:.3f}" if binary else f"|spearman| {strength:.2f}"
            add("high", "too-predictive", f"{col} alone predicts the target ({measure}); confirm it is known before the outcome")
        elif not math.isnan(miss) and miss >= 0.5:
            add("medium", "missingness-predictive",
                f"whether {col} is missing predicts the target (strength {miss:.2f}); often a process/leakage artifact")
    strength_table = pd.DataFrame(strengths, columns=["feature", "strength", "missingness_strength", "suspect_name"])
    strength_table = strength_table.sort_values("strength", ascending=False, na_position="last")
    for row in strength_table.head(10).itertuples():
        if row.suspect_name and not math.isnan(row.strength) and row.strength >= 0.3:
            add("medium", "suspect-name", f"{row.feature} has a post-outcome-sounding name and strength {row.strength:.2f}")

    # 6. Row order / ids carry the target.
    for col in [c for c in (id_col,) if c] + [c for c in features if re.search(r"(?i)(^id$|_id$|index|row)", c)]:
        if binary and pd.api.types.is_numeric_dtype(train[col]):
            strength = abs(auc(y_float.to_numpy(), train[col].to_numpy(dtype=float)) - 0.5) * 2
            if strength >= 0.2:
                add("medium", "id-predictive", f"id-like column {col} predicts the target (strength {strength:.2f}); "
                    "rows were likely sorted or assigned ids after the outcome")
    if binary:
        order_strength = abs(auc(y_float.to_numpy(), np.arange(len(train), dtype=float)) - 0.5) * 2
        if order_strength >= 0.2:
            add("low", "row-order", f"row position in train predicts the target (strength {order_strength:.2f}); "
                "shuffle before any positional split")

    # 7. Drift between splits.
    drift = []
    for col in features:
        value = psi(train[col], test[col])
        if not math.isnan(value):
            drift.append({"feature": col, "psi": value})
    drift_table = pd.DataFrame(drift, columns=["feature", "psi"]).sort_values("psi", ascending=False)
    for row in drift_table.itertuples():
        if row.psi > 0.25:
            add("medium", "split-drift", f"{row.feature} distribution differs between splits (PSI {row.psi:.2f})")
    if binary:
        prev_train, prev_test = float(y_float.mean()), float(test[target].astype(float).mean())
        if abs(prev_train - prev_test) > 0.1 * max(prev_train, 1e-9):
            add("low", "prevalence-shift", f"target rate train {prev_train:.3f} vs test {prev_test:.3f}")

    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: order[f["severity"]])
    return {"rows_train": len(train), "rows_test": len(test), "features": len(features), "binary": binary,
            "findings": findings, "single_feature_strength": strength_table.to_dict(orient="records"),
            "drift": drift_table.to_dict(orient="records")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--train")
    src.add_argument("--data")
    ap.add_argument("--test")
    ap.add_argument("--split", help="split column when using --data")
    ap.add_argument("--train-value", default="train")
    ap.add_argument("--target", required=True)
    ap.add_argument("--id")
    ap.add_argument("--group")
    ap.add_argument("--time")
    ap.add_argument("--text", nargs="*", default=[])
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--strength-threshold", type=float, default=0.8,
                    help="single-feature strength (|AUC-0.5|*2 or |spearman|) that counts as suspicious")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.train and not args.test:
        ap.error("--train needs --test")
    if args.data and not args.split:
        ap.error("--data needs --split")

    train, test = split_frames(args)
    result = run_checks(train, test, args.target, args.id, args.group, args.time, args.text, args.exclude,
                        args.strength_threshold)
    directory = out_dir(args.out, "leakage")
    md = [f"# Leakage check\n\ntrain {result['rows_train']} rows, test {result['rows_test']} rows, "
          f"{result['features']} features\n\n## Findings\n"]
    md += [f"- **{f['severity'].upper()}** [{f['rule']}] {f['message']}\n" for f in result["findings"]] or ["_none_\n"]
    md += ["\n## Strongest single features\n", md_table(pd.DataFrame(result["single_feature_strength"]), 15)]
    md += ["\n## Drift between splits (PSI)\n", md_table(pd.DataFrame(result["drift"]), 15)]
    json_path, md_path = write_report(directory, "leakage", result, "".join(md))
    print(f"leakage: {md_path}\njson: {json_path}")
    for f in result["findings"][:15]:
        print(f"  {f['severity'].upper():6} [{f['rule']}] {f['message']}")
    return 1 if any(f["severity"] == "high" for f in result["findings"]) else 0


if __name__ == "__main__":
    sys.exit(main())
