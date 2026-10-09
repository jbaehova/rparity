"""Clean-room statistical models with R-compatible public APIs."""
from .anova import Anova, anova
from .emm import EmmGrid, contrast, emmeans, emtrends, joint_tests, pairs, ref_grid, regrid
from .gls import corAR1, corARMA, corCompSymm, corSymm, gls, varExp, varIdent, varPower
from .lmm import glmer, lmer

__version__ = '0.1.0'
__all__ = [
    'Anova',
    'EmmGrid',
    'anova',
    'contrast',
    'corAR1',
    'corARMA',
    'corCompSymm',
    'corSymm',
    'emmeans',
    'emtrends',
    'glmer',
    'gls',
    'joint_tests',
    'lmer',
    'pairs',
    'ref_grid',
    'regrid',
    'varExp',
    'varIdent',
    'varPower',
]
