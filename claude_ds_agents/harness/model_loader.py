#!/usr/bin/env python3
"""Find and load trained models (sklearn, LightGBM, XGBoost, PyTorch) behind one predict().

  python model_loader.py scan [ROOT]                 # list model files; never executes them
  python model_loader.py inspect models/churn.txt    # load and describe one model
  python model_loader.py inspect model.pkl --trust-pickle
  python model_loader.py inspect net.pt --model-class mypkg.nets:ChurnNet --model-init '{"n_in": 12}'
  python model_loader.py inspect --factory mypkg.train:build_model   # model defined in code

scan identifies frameworks from file signatures, pickle opcodes and torch zip contents
without unpickling anything. Loading a pickle (joblib/pkl, or a torch file holding a whole
pickled module) runs code from the file, so it needs --trust-pickle: use it only for
artifacts your own project produced. Torch state_dicts load with weights_only=True.

Library use (model_diagnostics.py, agents' scratch scripts):
  from model_loader import load_model
  model = load_model("models/churn.txt")
  p = model.predict(frame)          # P(class 1) for binary classifiers, else the prediction
"""

from __future__ import annotations

import argparse
import importlib
import json
import pickletools
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

MODEL_SUFFIXES = {".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt", ".bin", ".txt", ".model", ".json",
                  ".ubj", ".xgb", ".bst", ".onnx", ".cbm", ".h5", ".keras", ".safetensors"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".dsguard", "site-packages"}
FRAMEWORK_MODULES = (("lightgbm", "lightgbm"), ("xgboost", "xgboost"), ("sklearn", "sklearn"),
                     ("torch", "torch"), ("catboost", "catboost"), ("statsmodels", "statsmodels"))


def _pickle_globals(data: bytes, limit: int = 200_000) -> list[str]:
    """Module.attr names referenced by a pickle stream, read with pickletools (no code runs)."""
    names, strings = [], []
    try:
        for op, arg, _ in pickletools.genops(data[:limit] if len(data) > limit else data):
            if op.name in ("GLOBAL", "INST") and isinstance(arg, str):
                names.append(arg.replace(" ", "."))
            elif op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE") and isinstance(arg, str):
                strings.append(arg)
            elif op.name == "STACK_GLOBAL" and len(strings) >= 2:
                names.append(f"{strings[-2]}.{strings[-1]}")
            if op.name == "STOP":
                break
    except Exception:  # truncated or compressed stream: report what was read
        pass
    return names


def _framework_from_globals(names: list[str]) -> str:
    for prefix, framework in FRAMEWORK_MODULES:
        if any(n.startswith(prefix + ".") for n in names):
            return framework
    return "unknown"


def identify(path: Path) -> dict:
    """Describe a model file from its bytes only."""
    info = {"path": str(path), "size_mb": round(path.stat().st_size / 1e6, 3), "framework": "unknown",
            "format": path.suffix.lstrip(".") or "?", "needs": [], "detail": ""}
    head = path.read_bytes()[:4096]
    suffix = path.suffix.lower()
    if zipfile.is_zipfile(path) and suffix in (".pt", ".pth", ".ckpt", ".bin"):
        with zipfile.ZipFile(path) as zf:
            pkl = next((n for n in zf.namelist() if n.endswith("data.pkl")), None)
            names = _pickle_globals(zf.read(pkl)) if pkl else []
        custom = sorted({n.rsplit(".", 1)[0] for n in names
                         if not n.startswith(("torch", "collections", "numpy", "builtins", "_codecs", "pytorch_lightning", "lightning"))})
        info["framework"] = "torch"
        if custom:
            info.update(format="torch-pickled-module", needs=["--trust-pickle"],
                        detail=f"pickled classes from: {', '.join(custom[:5])}")
        else:
            lightning = any(n.startswith(("pytorch_lightning", "lightning")) for n in names)
            info.update(format="torch-checkpoint" if lightning else "torch-state-dict", needs=["--model-class"],
                        detail="weights only; the model class must come from the project code")
    elif suffix in (".pkl", ".pickle", ".joblib") or head[:1] == b"\x80":
        names = _pickle_globals(path.read_bytes())
        info.update(format="pickle", needs=["--trust-pickle"])
        if names:
            info["framework"] = _framework_from_globals(names)
            info["detail"] = ", ".join(sorted({n for n in names if not n.startswith(("numpy", "builtins", "_codecs", "copyreg"))})[:6])
        else:
            info["detail"] = "compressed or unreadable pickle stream (joblib compression?)"
    elif head.startswith(b"tree") or (b"version=v" in head and b"num_class=" in head):
        info.update(framework="lightgbm", format="lightgbm-text")
    elif suffix in (".json", ".ubj", ".xgb", ".bst") and (b"learner" in head or suffix in (".ubj", ".xgb")):
        info.update(framework="xgboost", format=f"xgboost-{suffix.lstrip('.')}")
    elif suffix == ".onnx":
        info.update(framework="onnx", format="onnx", needs=["onnxruntime"])
    elif suffix == ".cbm":
        info.update(framework="catboost", format="catboost")
    elif suffix in (".h5", ".keras"):
        info.update(framework="keras", format=suffix.lstrip("."), detail="not supported by load_model; export predictions")
    elif suffix == ".safetensors":
        info.update(framework="torch", format="safetensors", needs=["--model-class"])
    else:
        return {}
    return info


def scan(root: Path) -> list[dict]:
    found = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES and not any(p in SKIP_DIRS for p in rel.parts):
            if path.suffix.lower() in (".txt", ".json", ".bin") and path.stat().st_size < 1024:
                continue
            info = identify(path)
            if info:
                info["path"] = rel.as_posix()
                found.append(info)
    return found


def _import(spec: str):
    module, _, attr = spec.partition(":")
    sys.path.insert(0, str(Path.cwd()))
    return getattr(importlib.import_module(module), attr)


@dataclass
class LoadedModel:
    """Uniform wrapper: predict(frame) -> 1-D array of P(class 1) or regression values."""
    framework: str
    kind: str                          # binary | multiclass | regression | unknown
    feature_names: list[str] | None
    raw: object
    _predict: object = field(repr=False, default=None)

    def predict(self, frame):
        import numpy as np
        data = frame[self.feature_names] if self.feature_names else frame
        return np.asarray(self._predict(data), dtype=float).reshape(len(frame), -1)[:, -1] \
            if self.kind == "binary" else np.asarray(self._predict(data), dtype=float)

    def describe(self) -> dict:
        return {"framework": self.framework, "kind": self.kind, "type": type(self.raw).__name__,
                "n_features": len(self.feature_names) if self.feature_names else None,
                "feature_names": self.feature_names}


def _wrap_lightgbm_booster(booster) -> LoadedModel:
    objective = str(booster.dump_model(num_iteration=1).get("objective", ""))
    kind = "binary" if objective.startswith(("binary", "cross_entropy")) else \
        "multiclass" if objective.startswith("multiclass") else "regression"
    return LoadedModel("lightgbm", kind, list(booster.feature_name()), booster, booster.predict)


def _wrap_xgboost_booster(booster) -> LoadedModel:
    import numpy as np
    import xgboost as xgb
    objective = json.loads(booster.save_config())["learner"]["objective"]["name"]
    kind = "binary" if objective.startswith("binary:") else "multiclass" if objective.startswith("multi:") else "regression"

    def predict(data):
        out = booster.predict(xgb.DMatrix(data))
        return 1 / (1 + np.exp(-out)) if objective == "binary:logitraw" else out
    return LoadedModel("xgboost", kind, booster.feature_names, booster, predict)


def wrap_object(obj) -> LoadedModel:
    """Wrap an in-memory model: sklearn estimator/pipeline, LightGBM/XGBoost sklearn API or Booster."""
    module = type(obj).__module__
    if module.startswith("lightgbm") and type(obj).__name__ == "Booster":
        return _wrap_lightgbm_booster(obj)
    if module.startswith("xgboost") and type(obj).__name__ == "Booster":
        return _wrap_xgboost_booster(obj)
    framework = next((f for prefix, f in FRAMEWORK_MODULES if module.startswith(prefix)), module.split(".")[0])
    names = getattr(obj, "feature_names_in_", None)
    names = [str(n) for n in names] if names is not None else None
    if hasattr(obj, "predict_proba"):
        classes = getattr(obj, "classes_", [0, 1])
        kind = "binary" if len(classes) == 2 else "multiclass"
        return LoadedModel(framework, kind, names, obj, obj.predict_proba)
    if hasattr(obj, "predict"):
        return LoadedModel(framework, "regression", names, obj, obj.predict)
    raise TypeError(f"{type(obj).__name__} has no predict/predict_proba")


def _load_torch(path: Path, info: dict, trust_pickle: bool, model_class: str | None, init_kwargs: dict,
                output: str, features: list[str] | None) -> LoadedModel:
    import numpy as np
    import torch
    if info["format"] == "torch-pickled-module":
        if not trust_pickle:
            raise PermissionError(f"{path} is a pickled torch module; pass --trust-pickle if your project produced it")
        net = torch.load(path, map_location="cpu", weights_only=False)
    else:
        if info["format"] == "safetensors":
            from safetensors.torch import load_file
            state = load_file(str(path))
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not model_class:
            raise ValueError(f"{path} holds weights only; pass --model-class module:Class (and --model-init JSON)")
        net = _import(model_class)(**init_kwargs)
        try:
            net.load_state_dict(state)
        except RuntimeError:
            prefix = next(iter(state)).split(".", 1)[0] + "."     # e.g. Lightning's "model." prefix
            net.load_state_dict({k.removeprefix(prefix): v for k, v in state.items()})
    net.eval()

    def predict(data):
        values = data.to_numpy(dtype="float32") if hasattr(data, "to_numpy") else np.asarray(data, "float32")
        with torch.no_grad():
            out = net(torch.from_numpy(values)).cpu().numpy()
        out = out.reshape(len(values), -1)
        if output == "logits":
            out = 1 / (1 + np.exp(-out))
        elif output == "softmax-logits":
            out = np.exp(out - out.max(axis=1, keepdims=True))
            out = out / out.sum(axis=1, keepdims=True)
        return out
    kind = "binary" if output in ("logits", "proba", "softmax-logits") else "regression"
    return LoadedModel("torch", kind, features, net, predict)


def load_model(path: str | None = None, *, trust_pickle: bool = False, model_class: str | None = None,
               init_kwargs: dict | None = None, factory: str | None = None, output: str = "value",
               features: list[str] | None = None) -> LoadedModel:
    """Load a model file, or call a project factory ('module:function') that returns a fitted model.

    output (torch only): value | proba | logits | softmax-logits, describing what the network returns.
    """
    if factory:
        model = wrap_object(_import(factory)())
        model.feature_names = features or model.feature_names
        return model
    p = Path(path)
    info = identify(p)
    if not info:
        raise ValueError(f"{path}: not a recognised model file")
    framework = info["framework"]
    if framework == "lightgbm" and info["format"] == "lightgbm-text":
        import lightgbm as lgb
        return _wrap_lightgbm_booster(lgb.Booster(model_file=str(p)))
    if framework == "xgboost" and info["format"].startswith("xgboost-"):
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(p))
        return _wrap_xgboost_booster(booster)
    if framework == "torch":
        return _load_torch(p, info, trust_pickle, model_class, init_kwargs or {}, output, features)
    if info["format"] == "pickle":
        if not trust_pickle:
            raise PermissionError(f"{path} is a pickle; loading runs code from it. Pass --trust-pickle if your project produced it")
        try:
            import joblib
            obj = joblib.load(p)
        except ImportError:
            import pickle
            with open(p, "rb") as fh:
                obj = pickle.load(fh)
        model = wrap_object(obj)
        model.feature_names = features or model.feature_names
        return model
    raise ValueError(f"{path}: {info['format']} models are recognised but not loadable here ({info['detail']})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan")
    s.add_argument("root", nargs="?", default=".")
    s.add_argument("--json", action="store_true")
    i = sub.add_parser("inspect")
    i.add_argument("path", nargs="?")
    add_load_args(i)
    args = ap.parse_args(argv)
    if args.command == "scan":
        found = scan(Path(args.root))
        if args.json:
            print(json.dumps(found, indent=2))
        else:
            for f in found:
                needs = f" needs {' '.join(f['needs'])}" if f["needs"] else ""
                print(f"{f['path']}  [{f['framework']}/{f['format']}, {f['size_mb']} MB]{needs}  {f['detail']}")
            print(f"{len(found)} model file(s)")
        return 0
    model = load_from_args(args, args.path)
    print(json.dumps(model.describe(), indent=2, default=str))
    return 0


def add_load_args(parser) -> None:
    parser.add_argument("--trust-pickle", action="store_true", help="allow unpickling (only your own artifacts)")
    parser.add_argument("--model-class", help="module:Class for torch state_dicts")
    parser.add_argument("--model-init", default="{}", help="JSON kwargs for --model-class")
    parser.add_argument("--factory", help="module:function returning a fitted model (model defined in code)")
    parser.add_argument("--output", default="value", choices=["value", "proba", "logits", "softmax-logits"],
                        help="what a torch network returns")


def load_from_args(args, path: str | None, features: list[str] | None = None) -> LoadedModel:
    return load_model(path, trust_pickle=args.trust_pickle, model_class=args.model_class,
                      init_kwargs=json.loads(args.model_init), factory=args.factory, output=args.output,
                      features=features)


if __name__ == "__main__":
    sys.exit(main())
