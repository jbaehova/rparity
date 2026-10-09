"""Clean-room statistical models with R-compatible public APIs."""
from .anova import Anova, anova
from .emm import EmmGrid, contrast, emmeans, emtrends, joint_tests, pairs, ref_grid, regrid
from .gam import GamResult, gam, gam_check
from .gls import corAR1, corARMA, corCompSymm, corSymm, gls, varExp, varIdent, varPower
from .lmm import glmer, lmer
from .tmb import GlmmTMBResult, glmmTMB

__version__ = '0.2.0'
__all__ = [
    'Anova',
    'EmmGrid',
    'GamResult',
    'GlmmTMBResult',
    'anova',
    'contrast',
    'corAR1',
    'corARMA',
    'corCompSymm',
    'corSymm',
    'emmeans',
    'emtrends',
    'gam',
    'gam_check',
    'glmer',
    'glmmTMB',
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
