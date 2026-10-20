"""Shared helpers for the DS harness scripts: loading tables, metrics, reports.

Needs only numpy and pandas so it runs in any project environment. Metrics are
implemented here (not imported from sklearn) to keep that dependency optional.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

TIME_UNITS = {"D": "day", "W": "week", "MS": "month"}


def load_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".feather":
        return pd.read_feather(path)
    if suffix in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in (".pkl", ".pickle"):
        return pd.read_pickle(path)  # only your own artifacts: pickle runs code
    sep = "\t" if suffix == ".tsv" else ","
    return pd.read_csv(path, sep=sep)


def out_dir(base: str | None, kind: str) -> Path:
    path = Path(base) if base else Path(".dsguard") / kind / datetime.now().strftime("%Y%m%d-%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    state = Path(".dsguard")
    if state.is_dir() and not (state / ".gitignore").exists():
        (state / ".gitignore").write_text("*\n")
    return path


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) or math.isinf(value) else round(float(value), 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def write_report(directory: Path, name: str, payload: dict, markdown: str) -> tuple[Path, Path]:
    json_path = directory / f"{name}.json"
    md_path = directory / f"{name}.md"
    json_path.write_text(json.dumps(to_jsonable(payload), indent=2, default=str))
    md_path.write_text(markdown)
    return json_path, md_path


def is_binary_target(y: pd.Series) -> bool:
    values = set(pd.unique(y.dropna()))
    return len(values) == 2 and values <= {0, 1}  # True/False and 0.0/1.0 hash equal to 0/1


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (ties get half credit)."""
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    mask = ~(np.isnan(y) | np.isnan(score))
    y, score = y[mask], score[mask]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    y = np.asarray(y, dtype=float)[order]
    if y.sum() == 0:
        return float("nan")
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Equal-width reliability table and expected calibration error."""
    frame = pd.DataFrame({"y": y, "p": p})
    frame["bin"] = pd.cut(frame["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    table = frame.groupby("bin", observed=True).agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
    table = table[table["n"] > 0].reset_index()
    table["bin"] = table["bin"].astype(str)
    ece = float((table["n"] * (table["mean_pred"] - table["observed"]).abs()).sum() / table["n"].sum())
    return table, ece


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    resid = y - pred
    ss_tot = ((y - y.mean()) ** 2).sum()
    metrics = {
        "n": len(y),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "r2": float(1 - (resid**2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        "bias_mean_residual": float(resid.mean()),
        "median_residual": float(np.median(resid)),
        "residual_skew": float(pd.Series(resid).skew()),
    }
    if (y != 0).all():
        metrics["mape"] = float(np.mean(np.abs(resid / y)))
    return metrics


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    _, ece = calibration_table(y, p)
    return {
        "n": len(y),
        "prevalence": float(y.mean()),
        "mean_prediction": float(p.mean()),
        "auc": auc(y, p),
        "average_precision": average_precision(y, p),
        "log_loss": log_loss(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece,
    }


def score(task: str, y: np.ndarray, pred: np.ndarray) -> float:
    """Single headline metric per task, higher is better (AUC, or negative RMSE)."""
    if task == "binary":
        return auc(y, pred)
    return -float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(pred, float)) ** 2)))


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population stability index of actual vs expected (numeric: quantile bins; else levels)."""
    expected, actual = expected.dropna(), actual.dropna()
    if expected.empty or actual.empty:
        return float("nan")
    if pd.api.types.is_numeric_dtype(expected) and expected.nunique() > bins:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = pd.cut(expected, edges).value_counts(normalize=True, sort=False)
        a = pd.cut(actual, edges).value_counts(normalize=True, sort=False)
    else:
        e = expected.astype(str).value_counts(normalize=True)
        a = actual.astype(str).value_counts(normalize=True)
        e, a = e.align(a, fill_value=0)
    e, a = e.clip(lower=1e-4), a.clip(lower=1e-4)
    return float(((a - e) * np.log(a / e)).sum())


def spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return float("nan")
    return float(frame["x"].rank().corr(frame["y"].rank()))


def oof_target_encoding(x: pd.Series, y: pd.Series, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold mean target per category, so a column cannot 'predict' itself."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, len(x))
    keys = x.astype(str).fillna("<NA>").to_numpy()
    yv = y.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    prior = float(np.nanmean(yv))
    for k in range(folds):
        train = fold != k
        means = pd.Series(yv[train]).groupby(keys[train]).mean()
        out[~train] = pd.Series(keys[~train]).map(means).fillna(prior).to_numpy()
    return out


def single_feature_strength(x: pd.Series, y: pd.Series, binary: bool) -> float:
    """How well one column alone predicts the target: AUC distance-from-chance scaled to [0, 1]
    for binary targets (|AUC-0.5|*2), |Spearman| for continuous targets."""
    numeric = pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)
    signal = x.to_numpy(dtype=float) if numeric else oof_target_encoding(x, y)
    if binary:
        value = auc(y.to_numpy(dtype=float), signal)
        return float("nan") if math.isnan(value) else abs(value - 0.5) * 2
    return abs(spearman(pd.Series(signal), y))


def time_frequency(times: pd.Series) -> str:
    span = (times.max() - times.min()).days
    if span > 180:
        return "MS"
    if span > 21:
        return "W"
    return "D"


def md_table(frame: pd.DataFrame, max_rows: int = 20, digits: int = 4) -> str:
    if frame.empty:
        return "_none_\n"
    frame = frame.head(max_rows)
    cols = list(frame.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in frame.to_numpy(dtype=object):
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if math.isnan(v) else f"{v:.{digits}g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
