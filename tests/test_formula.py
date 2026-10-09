"""Formula behavior contracts independent of any statistical fitting backend."""
import numpy as np
import pandas as pd
import pytest

from rparity.formula import build_fixed_design, build_random_design, parse_formula


@pytest.mark.parametrize('expression,count', [('(1|g)',1),('(x|g)',1),('(x||g)',2),('(1|g)+(1|h)',2),('(1|g/h)',2),('(0+x|g)',1)])
def test_random_designs(expression: str, count: int) -> None:
    data = pd.DataFrame({'y':[1,2,3,4], 'x':[0,1,2,3], 'g':['a','a','b','b'], 'h':[1,2,1,2]})
    parsed = parse_formula('y ~ x + ' + expression)
    fixed = build_fixed_design(parsed.fixed,data)
    blocks = build_random_design(parsed,fixed.data)
    assert len(blocks) == count
    assert np.allclose(fixed.X[:,1], data.x)
    for block in blocks:
        assert np.allclose(block.Z.sum(axis=1), block.values.sum(axis=1))


def test_missing_random_covariate_omitted() -> None:
    data = pd.DataFrame({'y':[1,2,3], 'x':[1,np.nan,3], 'g':['a','b','c']})
    design=build_fixed_design('y ~ 1 + (x|g)',data)
    assert len(design.y)==2


def test_sum_contrast_and_offset() -> None:
    d=pd.DataFrame({'y':[1,2,3,4], 'a':['a','a','b','b'], 'o':[0,0,1,1]})
    parsed=parse_formula('y ~ a + offset(o) + (1|a)')
    assert parsed.offsets==['o']
    x=build_fixed_design(parsed.fixed,d,contrasts='sum').X
    assert np.allclose(x[:,1],[1,1,-1,-1])


def test_polars_input() -> None:
    pl=pytest.importorskip('polars')
    design=build_fixed_design('y ~ x',pl.DataFrame({'y':[1,2,3], 'x':[0,1,2]}))
    assert design.X.shape==(3,2)


@pytest.mark.parametrize('formula',['y x','y ~ (x|g','y ~ (|g)','y ~ x | g'])
def test_invalid_formula(formula: str) -> None:
    with pytest.raises(ValueError):
        parse_formula(formula)
