"""Mathematical spline and model-matrix invariants independent of an oracle."""
import numpy as np
import pandas as pd
import pytest

from rparity.gam._basis import center_basis, spline_basis, tensor_basis
from rparity.gam._formula import build_design, parse_gam_formula


@pytest.fixture
def data():
    rng = np.random.default_rng(18)
    n = 72
    return pd.DataFrame({
        "y": rng.normal(size=n), "x": np.sort(rng.uniform(-1, 2, n)),
        "z": rng.uniform(-2, 1, n), "w": rng.uniform(0.4, 1.6, n),
        "g": pd.Categorical(np.tile(["a", "b", "c"], n // 3)),
    })


@pytest.mark.parametrize("kind", ["tp", "cr", "cs", "ps"])
def test_splines_have_required_space_and_psd_penalty(data, kind):
    x = data.x.to_numpy()
    basis = spline_basis(x, kind, 8)
    X, S = basis.matrix(x), basis.S[0]
    assert X.shape == (len(x), 8)
    assert np.linalg.matrix_rank(X) == 8
    assert np.linalg.eigvalsh(S).min() >= -np.linalg.norm(S) * 1e-12
    for y in [np.ones(len(x)), x]:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        np.testing.assert_allclose(X @ beta, y, atol=1e-10)
        if kind != "cs":
            assert abs(beta @ S @ beta) < np.linalg.norm(S) * np.linalg.norm(beta) ** 2 * 1e-10
    assert np.linalg.matrix_rank(S) == (8 if kind == "cs" else 6)


@pytest.mark.parametrize("kind", ["tp", "cr", "cs", "ps"])
def test_centering_absorbs_one_dimension_without_changing_penalty(data, kind):
    x = data.x.to_numpy()
    raw = spline_basis(x, kind, 8)
    centered = center_basis(raw, x)
    assert centered.k == 7
    np.testing.assert_allclose(centered.matrix(x).mean(0), 0, atol=1e-12)
    change = np.linalg.lstsq(raw.matrix(x), centered.matrix(x), rcond=None)[0]
    np.testing.assert_allclose(change.T @ raw.S[0] @ change, centered.S[0], atol=1e-8)


@pytest.mark.parametrize("kind", ["cr", "cs", "ps"])
def test_cubic_splines_extrapolate_linearly(data, kind):
    x = data.x.to_numpy()
    basis = spline_basis(x, kind, 7)
    points = np.array([x.min() - 3, x.min() - 2, x.min() - 1,
                       x.max() + 1, x.max() + 2, x.max() + 3])
    evaluation = basis.matrix(points)
    np.testing.assert_allclose(np.diff(evaluation[:3], n=2, axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(np.diff(evaluation[3:], n=2, axis=0), 0, atol=1e-12)


def test_tensor_penalties_apply_one_marginal_at_a_time(data):
    a = spline_basis(data.x.to_numpy(), "cr", 5)
    b = spline_basis(data.z.to_numpy(), "cr", 4)
    tensor = tensor_basis([a, b], "te")
    aa = np.array([0.3, -0.2, 0.4, 1, 0.2])
    bb = np.array([1, -0.1, 0.5, 0.2])
    beta = np.kron(aa, bb)
    np.testing.assert_allclose(tensor.matrix(data[["x", "z"]].to_numpy()) @ beta,
                               (a.matrix(data.x.to_numpy()) @ aa)
                               * (b.matrix(data.z.to_numpy()) @ bb), atol=1e-12)
    np.testing.assert_allclose(beta @ tensor.S[0] @ beta,
                               (aa @ a.S[0] @ aa) * (bb @ bb), atol=1e-10)
    np.testing.assert_allclose(beta @ tensor.S[1] @ beta,
                               (bb @ b.S[0] @ bb) * (aa @ aa), atol=1e-10)


@pytest.mark.parametrize("formula", [
    'y ~ z + s(x, k=8)', 'y ~ s(x, bs="cr", k=7)',
    'y ~ s(x, bs="cs", k=6)', 'y ~ s(x, bs="ps", k=6)',
    'y ~ s(g, bs="re")', 'y ~ s(x, g, bs="re")',
    'y ~ te(x,z,k=c(5,4))', 'y ~ ti(x) + ti(z) + ti(x,z)',
    'y ~ g + s(x,by=g)', 'y ~ s(x,by=w)',
    'y ~ s(log(w),k=7) + offset(z)', 'y ~ 0 + s(x,z,k=12)',
])
def test_prediction_reuses_training_design(data, formula):
    design = build_design(formula, data)
    np.testing.assert_allclose(design.predict(data), design.X, atol=1e-12)
    np.testing.assert_allclose(design.get_model_matrix(data.iloc[:5]), design.X[:5], atol=1e-12)
    assert design.X.shape[1] == len(design.coef_names)
    assert all(s.shape == (design.X.shape[1],) * 2 for s in design.S)


def test_factor_by_masks_levels_after_global_centering(data):
    design = build_design('y ~ g + s(x,by=g,bs="cr",k=7)', data)
    assert len(design.smooths) == 3
    for smooth in design.smooths:
        term = design.X[:, smooth.indices]
        assert np.all(term[data.g != smooth.level] == 0)
        # Centering concerns the underlying function across all x values,
        # rather than the subset for a particular factor level.
        basis = smooth.basis
        np.testing.assert_allclose(basis.matrix(data.x.to_numpy()).mean(0), 0, atol=1e-12)


def test_numeric_by_preserves_constant_function_for_varying_coefficient(data):
    design = build_design('y ~ s(x,by=w,bs="cr",k=7)', data)
    smooth = design.smooths[0]
    assert len(smooth.indices) == 7
    beta = np.linalg.lstsq(design.X[:, smooth.indices], data.w, rcond=None)[0]
    np.testing.assert_allclose(design.X[:, smooth.indices] @ beta, data.w, atol=1e-12)


def test_random_effects_unknown_level_has_zero_design(data):
    design = build_design('y ~ s(g,bs="re")', data)
    new = data.iloc[:2].copy()
    new["g"] = ["unobserved", "a"]
    prediction = design.predict(new)
    assert np.all(prediction[0, design.smooths[0].indices] == 0)
    assert prediction[1, design.smooths[0].indices].sum() == 1


def test_missing_smooth_rows_and_offsets_remain_aligned(data):
    data.loc[3, "x"] = np.nan
    design = build_design('y ~ s(x) + offset(z)', data)
    assert len(design.y) == len(data) - 1
    np.testing.assert_allclose(design.offset, data.dropna().z)
    np.testing.assert_allclose(design.prediction_offset(data.dropna()), design.offset)


def test_cbind_response_retains_success_failure_columns(data):
    data["success"] = np.arange(len(data)) % 5
    data["failure"] = 5 - data.success
    design = build_design('cbind(success,failure) ~ s(x)', data)
    np.testing.assert_array_equal(design.y, data[["success", "failure"]].to_numpy())


def test_formula_parser_retains_literal_vectors_and_intercept():
    fixed, specs = parse_gam_formula('y ~ 0 + z + te(x,z,k=c(4,5),bs=c("cr","ps"))')
    assert fixed == 'y ~ 0 + z'
    assert specs[0].k == (4, 5)
    assert specs[0].bs == ("cr", "ps")


@pytest.mark.parametrize("formula", [
    'y ~ s(x, bs="unknown")', 'y ~ s(x,k=999)',
    'y ~ te(x,z,k=c(4,5,6))', 'y ~ s(x, k=open("file"))',
    'y ~ s(__import__("os").system("touch should_not_exist"))',
])
def test_invalid_or_unsafe_smooth_specifications_raise(data, formula):
    with pytest.raises((ValueError, TypeError)):
        build_design(formula, data)
