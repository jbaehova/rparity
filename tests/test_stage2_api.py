"""Public Stage 2 workflows, input interoperability, and component semantics."""

import ast
import inspect
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit

import rparity
from rparity import Anova, emmeans, gam, gam_check, glmmTMB


def _frame(seed=412, n=96):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    z = rng.uniform(0.2, 1.2, n)
    a = np.where(np.arange(n) % 2, "B", "A")
    return pd.DataFrame({"x": x, "z": z, "a": a,
                         "y": 0.8 + np.sin(2 * x) + rng.normal(0, 0.3, n)})


@pytest.mark.parametrize("family", ["gaussian", "binomial", "poisson", "Gamma"])
@pytest.mark.parametrize("method", ["REML", "ML", "GCV.Cp"])
def test_gam_public_family_method_lifecycle(family, method):
    frame = _frame()
    rng = np.random.default_rng(392)
    eta = 0.2 + 0.5 * np.sin(2 * frame.x.to_numpy())
    if family == "binomial":
        frame.y = rng.binomial(1, expit(eta))
    elif family == "poisson":
        frame.y = rng.poisson(np.exp(eta))
    elif family == "Gamma":
        frame.y = rng.gamma(5, np.exp(eta) / 5)
    model = gam("y ~ s(x,bs='cr',k=5)", frame, family=family, method=method,
                link="log" if family == "Gamma" else None)
    assert isinstance(model, rparity.GamResult)
    assert model.method == method
    coefficients = model.coef()
    assert isinstance(coefficients, pd.Series)
    assert model.vcov().index.equals(coefficients.index)
    assert model.vcov().columns.equals(coefficients.index)
    assert len(model.edf) == len(coefficients)
    assert_allclose(model.summary_tables()["smooth"]["edf"], model.smooth_edf)
    new = frame.iloc[[7, 3, 15]].drop(columns="y")
    prediction = model.predict(new, type="response", se_fit=True)
    assert prediction["fit"].index.equals(new.index)
    assert prediction["se.fit"].index.equals(new.index)
    assert np.isfinite(prediction["fit"]).all()
    assert (prediction["se.fit"] > 0).all()
    effect = model.partial_effects("s(x)", newdata=new)
    terms = model.predict(new, type="terms", terms=["s(x)"], se_fit=True)
    assert list(terms["fit"]) == ["s(x)"]
    assert_allclose(effect["fit"], terms["fit"]["s(x)"])
    assert_allclose(effect["se"], terms["se.fit"]["s(x)"])
    summary = model.summary()
    sections = ["Family:", "Parametric coefficients:",
                "Approximate significance of smooth terms:"]
    positions = [summary.index(section) for section in sections]
    assert positions == sorted(positions)


@pytest.mark.parametrize("fitter,formula,kwargs", [
    (gam, "y ~ a+s(x,bs='cr',k=5)", {"method": "REML", "sp": [1.2]}),
    (glmmTMB, "y ~ x+a", {"family": "gaussian"}),
])
def test_stage2_polars_input_and_prediction_match_pandas(fitter, formula, kwargs):
    frame = _frame()
    polars = pl.DataFrame(frame.to_dict(orient="list"))
    model = fitter(formula, frame, **kwargs)
    alternative = fitter(formula, polars, **kwargs)
    assert isinstance(alternative.data, pd.DataFrame)
    assert_allclose(alternative.coef() if fitter is gam else alternative.fixef()["cond"],
                    model.coef() if fitter is gam else model.fixef()["cond"],
                    rtol=1e-10, atol=1e-10)
    new = frame.iloc[:8].drop(columns="y")
    pl_new = pl.DataFrame(new.to_dict(orient="list"))
    assert_allclose(alternative.predict(pl_new, type="response"),
                    model.predict(new, type="response"), rtol=1e-10, atol=1e-10)


def test_factor_and_numeric_by_prediction_and_partial_effects():
    frame = _frame()
    frame.a = pd.Categorical(frame.a, categories=["A", "B"])
    factor = gam("y ~ a+s(x,by=a,bs='cr',k=5)", frame, sp=[1, 1])
    labels = list(factor.smooth_edf.index)
    assert len(labels) == 2
    new = pd.DataFrame({"x": [0.2, 0.2], "a": pd.Categorical(["A", "B"],
                       categories=["A", "B"])})
    terms = factor.predict(new, type="terms")
    for label in labels:
        active_level = label.rsplit("a", 1)[-1]
        inactive_row = 1 if active_level == "A" else 0
        assert terms[label].iloc[inactive_row] == pytest.approx(0, abs=1e-14)
    effects = factor.partial_effects(newdata=new)
    assert set(effects) == set(labels)
    for label in labels:
        assert_allclose(effects[label]["fit"], terms[label])
        level = label.rsplit("a", 1)[-1]
        automatic = factor.partial_effects(label, n=9)
        assert set(automatic.a) == {level}
    numeric = gam("y ~ z+s(x,by=z,bs='cr',k=5)", frame, sp=[1])
    zero_multiplier = pd.DataFrame({"x": [-0.5, 0.5], "z": [0.0, 0.0]})
    assert_allclose(numeric.predict(zero_multiplier, type="terms")["s(x):z"], 0,
                    atol=1e-14)


def test_gam_check_reproducible_without_global_rng_mutation():
    model = gam("y ~ s(x,bs='cr',k=5)", _frame(), sp=[1])
    np.random.seed(735)
    expected = np.random.random(4)
    np.random.seed(735)
    first = gam_check(model, seed=17, k_rep=12, k_sample=50)
    second = model.check(seed=17, k_rep=12, k_sample=50)
    pd.testing.assert_frame_equal(first["k.check"], second["k.check"])
    assert_allclose(np.random.random(4), expected, atol=0)
    assert {"k'", "edf", "k-index", "p-value"} == set(first["k.check"].columns)
    assert 0 <= first["k.check"]["p-value"].iloc[0] <= 1
    assert len(first["residuals"]) == model.nobs


@pytest.mark.parametrize("family", ["poisson", "nbinom1", "nbinom2", "binomial", "beta",
                                    "gaussian"])
def test_tmb_public_family_and_empty_component_contract(family):
    frame = _frame(seed=883, n=180)
    rng = np.random.default_rng(268)
    eta = 0.4 + 0.3 * frame.x.to_numpy()
    if family == "poisson":
        frame.y = rng.poisson(np.exp(eta))
    elif family in {"nbinom1", "nbinom2"}:
        mu = np.exp(eta)
        size = mu / 0.8 if family == "nbinom1" else np.repeat(3.0, len(mu))
        frame.y = rng.negative_binomial(size, size / (size + mu))
    elif family == "binomial":
        frame.y = rng.binomial(1, expit(eta))
    elif family == "beta":
        mu = expit(eta)
        frame.y = rng.beta(8 * mu, 8 * (1 - mu))
    model = glmmTMB("y ~ x", frame, family=family)
    assert isinstance(model, rparity.GlmmTMBResult)
    components = model.fixef()
    assert set(components) == {"cond", "zi", "disp"}
    assert components["zi"].empty
    assert components["disp"].empty == (family in {"poisson", "binomial"})
    assert model.vcov("zi").shape == (0, 0)
    assert model.ranef() == {"cond": {}, "zi": {}}
    assert model.VarCorr() == {"cond": {}, "zi": {}}
    assert_allclose(model.predict(type="zprob"), 0, atol=0)
    assert_allclose(model.predict(type="cond"), model.predict(type="response"), atol=0)
    assert model.AIC() > -2 * model.logLik()
    assert "Conditional model:" in model.summary()
    with pytest.raises(ValueError, match="no estimated zi component"):
        emmeans(model, "x", component="zi")
    with pytest.raises(ValueError, match="no estimated zi component"):
        Anova(model, component="zi")


@pytest.fixture(scope="module")
def distributional_model():
    frame = _frame(seed=946, n=400)
    frame.a = pd.Categorical(frame.a, categories=["A", "B"])
    rng = np.random.default_rng(669)
    mu = np.exp(0.9 + 0.25 * frame.x + 0.4 * (frame.a == "B"))
    phi = np.exp(1.2 + 0.6 * (frame.a == "B"))
    response = rng.negative_binomial(phi, phi / (phi + mu))
    response[rng.random(len(frame)) < expit(-0.7 + 0.7 * (frame.a == "B"))] = 0
    frame.y = response
    return glmmTMB("y ~ x+a", frame, family="nbinom2", ziformula="~a", dispformula="~a")


@pytest.mark.parametrize("component,prediction_type", [
    ("cond", "cond"), ("zi", "zprob"), ("disp", "disp"),
])
def test_tmb_component_emmeans_anova_workflow(distributional_model, component, prediction_type):
    model = distributional_model
    options = {"at": {"x": [0.0]}} if component == "cond" else {}
    means = emmeans(model, "a", component=component, **options)
    prediction_data = means.grid.copy()
    prediction_data["x"] = 0.0
    reference = model.predict(prediction_data, type=prediction_type, re_form="NA")
    response = means.summary(type="response")
    estimate_column = next(name for name in ("response", "prob", "rate", "emmean")
                           if name in response)
    assert_allclose(response[estimate_column], reference, rtol=1e-12, atol=1e-12)
    assert np.isinf(response.df).all()
    pairwise = means.pairs(adjust="none")
    contrast_table = pairwise.summary(infer=True)
    assert len(contrast_table) == 1
    factor_coefficient = model.fixef()[component].iloc[-1]
    factor_variance = model.vcov(component).iloc[-1, -1]
    assert abs(contrast_table.estimate.iloc[0]) == pytest.approx(abs(factor_coefficient))
    assert contrast_table.SE.iloc[0] == pytest.approx(np.sqrt(factor_variance))
    for kind in (2, 3):
        table = Anova(model, type=kind, component=component)
        assert table.attrs["component"] == component
        assert table.loc["a", "Chisq"] == pytest.approx(factor_coefficient**2 / factor_variance)


def test_tmb_distributional_response_and_joint_covariance(distributional_model):
    model = distributional_model
    new = pd.DataFrame({"x": [-0.4, 0.5], "a": pd.Categorical(["A", "B"],
                       categories=["A", "B"])})
    conditional = model.predict(new, type="cond")
    probability = model.predict(new, type="zprob")
    assert ((probability > 0) & (probability < 1)).all()
    assert_allclose(model.predict(new, type="response"), conditional * (1 - probability),
                    rtol=1e-12, atol=1e-12)
    covariance = model.vcov(full=True)
    assert covariance.shape[0] == sum(map(len, model.fixef().values()))
    assert np.min(np.linalg.eigvalsh(covariance)) > 0
    for section in ["Conditional model:", "Zero-inflation model:", "Dispersion model:"]:
        assert section in model.summary()


def test_tmb_random_slope_predictions_use_modes_and_new_levels():
    rng = np.random.default_rng(493)
    group = np.repeat(np.arange(8), 12)
    x = np.tile(np.linspace(-1, 1, 12), 8)
    intercept = np.array([-1.2, -0.8, -0.4, -0.1, 0.2, 0.5, 0.8, 1.1])
    slope = np.array([-0.7, 0.4, -0.1, 0.5, 0.6, -0.3, 0.2, -0.6])
    frame = pd.DataFrame({"x": x, "g": [f"g{v}" for v in group],
                          "y": 0.4 + 0.2 * x + intercept[group] + slope[group] * x
                          + rng.normal(0, 0.15, len(x))})
    model = glmmTMB("y ~ x+(x|g)", frame, family="gaussian")
    modes = model.ranef()["cond"]["g"]
    assert set(modes.columns) == {"(Intercept)", "x"}
    assert modes.attrs["condVar"].shape == (2, 2, 8)
    covariance = model.VarCorr()["cond"]["g"]
    assert covariance.shape == (2, 2)
    assert np.min(np.linalg.eigvalsh(covariance)) >= 0
    new = frame.iloc[[2, 14, 26]].drop(columns="y")
    random_effect = np.asarray([modes.loc[row.g, "(Intercept)"] + row.x * modes.loc[row.g, "x"]
                               for row in new.itertuples()])
    assert_allclose(model.predict(new, type="link"),
                    model.predict(new, type="link", re_form="NA") + random_effect,
                    atol=1e-12)
    new.g = "unseen"
    with pytest.raises(ValueError, match="New group level"):
        model.predict(new)
    assert_allclose(model.predict(new, allow_new_levels=True), model.predict(new, re_form="NA"))


def test_public_stage2_workflows_never_launch_external_processes(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("A normal rparity API launched an external process")

    monkeypatch.setattr(subprocess, "Popen", blocked)
    monkeypatch.setattr(subprocess, "run", blocked)
    frame = _frame()
    additive = gam("y ~ s(x,bs='cr',k=5)", frame, sp=[1])
    additive.summary()
    additive.partial_effects(n=7)
    gam_check(additive, k_rep=5)
    mixed = glmmTMB("y ~ x+a", frame)
    mixed.summary()
    emmeans(mixed, "a").summary()
    Anova(mixed)


def test_stage2_runtime_has_no_r_integration_imports():
    source_root = Path(inspect.getfile(rparity)).parent
    forbidden = {"rpy2", "subprocess", "oracle"}
    for module in ("gam", "tmb"):
        for path in (source_root / module).glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [entry.name.split(".")[0] for entry in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [(node.module or "").split(".")[0]]
                else:
                    continue
                assert not forbidden.intersection(names), str(path)
