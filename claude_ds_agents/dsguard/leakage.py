"""Static leakage patterns in ML code: fitting preprocessing before the split, fitting on test.

Notebook cells are analysed in order as one program, since that is how they run.
Data-level leakage (duplicates, future timestamps, too-predictive columns) needs the
data itself: see harness/leakage_check.py.
"""

from __future__ import annotations

import ast
import re

from .core import CheckResult, Context, call_name, kwarg, python_sources

PREPROCESSORS = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer", "QuantileTransformer",
    "PowerTransformer", "OneHotEncoder", "OrdinalEncoder", "LabelEncoder", "TargetEncoder", "SimpleImputer",
    "KNNImputer", "IterativeImputer", "PCA", "TruncatedSVD", "TfidfVectorizer", "CountVectorizer",
    "HashingVectorizer", "SelectKBest", "SelectPercentile", "RFE", "RFECV", "VarianceThreshold",
    "KBinsDiscretizer", "PolynomialFeatures", "SelectFromModel",
}
RESAMPLERS = {"SMOTE", "ADASYN", "RandomOverSampler", "RandomUnderSampler", "BorderlineSMOTE", "SMOTENC"}
# Calls that actually separate train from evaluation data. Building a KFold object does not;
# calling .split() on one does (tracked via CV_CLASSES so str.split / re.split do not count).
SPLIT_CALLS = {"train_test_split", "cross_val_score", "cross_validate", "cross_val_predict", "GridSearchCV",
               "RandomizedSearchCV", "HalvingGridSearchCV", "HalvingRandomSearchCV"}
CV_CLASSES = {"KFold", "StratifiedKFold", "GroupKFold", "StratifiedGroupKFold", "TimeSeriesSplit", "ShuffleSplit",
              "StratifiedShuffleSplit", "GroupShuffleSplit", "RepeatedKFold", "RepeatedStratifiedKFold", "LeaveOneGroupOut"}
HOLDOUT_NAME = re.compile(r"(?i)(^|_)(test|te|tst|val|valid|validation|holdout|oot)(_|$|\d)")
FIT_METHODS = {"fit", "fit_transform"}


def _ordered_nodes(tree: ast.Module):
    """Statement-ordered walk (ast.walk is breadth-first, which scrambles line order)."""
    return sorted((n for n in ast.walk(tree) if hasattr(n, "lineno")), key=lambda n: (n.lineno, n.col_offset))


def _arg_names(call: ast.Call) -> list[str]:
    names = []
    for arg in call.args:
        if isinstance(arg, ast.Name):
            names.append(arg.id)
        elif isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
            names.append(arg.value.id)
    return names


def run(ctx: Context) -> CheckResult:
    result = CheckResult("leakage")
    programs: dict[str, list] = {}
    for src in python_sources(ctx):
        if src.tree is not None:
            programs.setdefault(src.path, []).append(src)

    for path, sources in programs.items():
        bound: dict[str, str] = {}  # variable -> preprocessor/resampler class
        split_seen = False
        for src in sources:
            for node in _ordered_nodes(src.tree):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                    cls = call_name(node.value.func)
                    if cls in PREPROCESSORS | RESAMPLERS | CV_CLASSES:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                bound[target.id] = cls
                if not isinstance(node, ast.Call):
                    continue
                name = call_name(node.func)
                if name in SPLIT_CALLS:
                    split_seen = True
                    continue
                if not isinstance(node.func, ast.Attribute):
                    continue
                receiver = node.func.value
                if name == "split" and (
                    (isinstance(receiver, ast.Name) and bound.get(receiver.id) in CV_CLASSES)
                    or (isinstance(receiver, ast.Call) and call_name(receiver.func) in CV_CLASSES)
                ):
                    split_seen = True
                    continue
                cls = None
                if isinstance(receiver, ast.Name):
                    cls = bound.get(receiver.id)
                    cls = cls if cls in PREPROCESSORS | RESAMPLERS else None
                elif isinstance(receiver, ast.Call):
                    cls = call_name(receiver.func) if call_name(receiver.func) in PREPROCESSORS | RESAMPLERS else None
                args = _arg_names(node)

                if name in FIT_METHODS | {"fit_resample"} and any(HOLDOUT_NAME.search(a) for a in args):
                    holdout = next(a for a in args if HOLDOUT_NAME.search(a))
                    result.add("fit-on-holdout", "warn", path,
                               f".{name}() on '{holdout}': fitting on evaluation data leaks it into the model",
                               node.lineno, src.cell)
                    continue
                if not cls or split_seen:
                    continue
                if name == "fit_resample":
                    result.add("resample-before-split", "warn", path,
                               f"{cls}.fit_resample() before the train/test split puts synthetic copies of test rows in train",
                               node.lineno, src.cell)
                elif name in FIT_METHODS:
                    result.add("fit-before-split", "warn", path,
                               f"{cls}.{name}() runs before any train/test split or CV: test statistics leak into "
                               "training; fit inside a Pipeline on the training fold", node.lineno, src.cell)

            for node in ast.walk(src.tree):
                if not isinstance(node, ast.Call):
                    continue
                name = call_name(node.func)
                if name == "shift":
                    periods = node.args[0] if node.args else (kwarg(node, "periods").value if kwarg(node, "periods") else None)
                    if isinstance(periods, ast.UnaryOp) and isinstance(periods.op, ast.USub):
                        result.add("future-shift", "info", path,
                                   "shift(-n) reads future rows: fine for building a target, leakage if used as a feature",
                                   node.lineno, src.cell)
                elif name in ("bfill", "backfill") or (
                    name == "fillna" and (m := kwarg(node, "method")) and getattr(m.value, "value", None) in ("bfill", "backfill")
                ):
                    result.add("backfill", "info", path, "backfill copies later values into earlier rows; leakage in time series",
                               node.lineno, src.cell)
    return result
