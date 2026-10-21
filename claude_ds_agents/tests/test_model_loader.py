"""model_loader against real frameworks; each block skips when its framework is not installed."""

import json
import pickle
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))

import model_loader  # noqa: E402

RNG = np.random.default_rng(5)


def _data(n=600):
    X = pd.DataFrame({"a": RNG.normal(size=n), "b": RNG.normal(size=n), "c": RNG.normal(size=n)})
    y = (X["a"] + 0.5 * X["b"] + RNG.normal(0, 0.5, n) > 0).astype(int)
    return X, y


def test_scan_identifies_without_unpickling(tmp_path):
    class Boom:
        def __reduce__(self):
            return (exec, ("raise SystemExit('executed!')",))
    (tmp_path / "evil.pkl").write_bytes(pickle.dumps(Boom()))
    (tmp_path / "lgb.txt").write_text("tree\nversion=v4\nnum_class=1\n" + "x" * 2000)
    (tmp_path / "xgb.json").write_text(json.dumps({"learner": {"objective": "binary:logistic"}, "pad": "x" * 2000}))
    (tmp_path / "notes.txt").write_text("just text " * 300)
    found = {f["path"]: f for f in model_loader.scan(tmp_path)}
    assert set(found) == {"evil.pkl", "lgb.txt", "xgb.json"}
    assert found["evil.pkl"]["needs"] == ["--trust-pickle"] and "builtins" not in found["evil.pkl"]["detail"]
    assert found["lgb.txt"]["framework"] == "lightgbm" and found["xgb.json"]["framework"] == "xgboost"
    with pytest.raises(PermissionError):
        model_loader.load_model(str(tmp_path / "evil.pkl"))


def test_sklearn_pickle_and_factory(tmp_path, monkeypatch):
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    X, y = _data()
    clf = LogisticRegression().fit(X, y)
    path = tmp_path / "clf.pkl"
    path.write_bytes(pickle.dumps(clf))
    info = model_loader.identify(path)
    assert info["framework"] == "sklearn"
    model = model_loader.load_model(str(path), trust_pickle=True)
    assert model.kind == "binary" and model.feature_names == ["a", "b", "c"]
    assert np.allclose(model.predict(X), clf.predict_proba(X)[:, 1])

    monkeypatch.chdir(tmp_path)
    (tmp_path / "factory_mod.py").write_text(
        "import pickle\ndef build():\n    return pickle.load(open('clf.pkl', 'rb'))\n")
    via_code = model_loader.load_model(factory="factory_mod:build")
    assert np.allclose(via_code.predict(X), clf.predict_proba(X)[:, 1])
    assert sklearn


def test_lightgbm_text_model(tmp_path):
    lgb = pytest.importorskip("lightgbm")
    X, y = _data()
    booster = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, y), num_boost_round=10)
    path = tmp_path / "churn.txt"
    booster.save_model(str(path))
    model = model_loader.load_model(str(path))
    assert model.framework == "lightgbm" and model.kind == "binary" and model.feature_names == ["a", "b", "c"]
    p = model.predict(X[["c", "b", "a"]])              # column order is fixed by feature_names
    assert np.allclose(p, booster.predict(X)) and 0 <= p.min() <= p.max() <= 1


def test_xgboost_json_model(tmp_path):
    xgb = pytest.importorskip("xgboost")
    X, y = _data()
    booster = xgb.train({"objective": "binary:logistic"}, xgb.DMatrix(X, y), num_boost_round=10)
    path = tmp_path / "model.json"
    booster.save_model(str(path))
    model = model_loader.load_model(str(path))
    assert model.framework == "xgboost" and model.kind == "binary"
    assert np.allclose(model.predict(X), booster.predict(xgb.DMatrix(X)))

    reg = xgb.train({"objective": "reg:squarederror"}, xgb.DMatrix(X, X["a"]), num_boost_round=5)
    reg.save_model(str(tmp_path / "reg.ubj"))
    assert model_loader.load_model(str(tmp_path / "reg.ubj")).kind == "regression"


def test_torch_state_dict_and_pickled_module(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "nets.py").write_text(
        "import torch\nclass Net(torch.nn.Module):\n"
        "    def __init__(self, n_in=3):\n        super().__init__()\n        self.lin = torch.nn.Linear(n_in, 1)\n"
        "    def forward(self, x):\n        return self.lin(x)\n")
    sys.path.insert(0, str(tmp_path))
    import nets
    net = nets.Net()
    torch.save(net.state_dict(), tmp_path / "w.pt")
    torch.save(net, tmp_path / "full.pt")
    assert model_loader.identify(tmp_path / "w.pt")["format"] == "torch-state-dict"
    assert model_loader.identify(tmp_path / "full.pt")["format"] == "torch-pickled-module"
    X, _ = _data(50)
    model = model_loader.load_model(str(tmp_path / "w.pt"), model_class="nets:Net", output="logits",
                                    features=["a", "b", "c"])
    expected = torch.sigmoid(net(torch.tensor(X.to_numpy(dtype="float32")))).detach().numpy().ravel()
    assert np.allclose(model.predict(X), expected, atol=1e-6)
    with pytest.raises(PermissionError):
        model_loader.load_model(str(tmp_path / "full.pt"))
