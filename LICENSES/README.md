# Bundled numerical library notices

rparity wheels vendor the unmodified numerical libraries from
[scipy-openblas32 0.3.34.237.0](https://pypi.org/project/scipy-openblas32/0.3.34.237.0/).
The project supports this build-time vendoring use. It is not a runtime package dependency.

The full upstream notices are retained verbatim in `scipy-openblas32-LICENSE.txt`
and in each wheel under `rparity/_openblas_libs/licenses/LICENSE.txt`. They cover
OpenBLAS and LAPACK under BSD terms. Compiler support libraries retain the GCC
runtime exception and the libquadmath LGPL terms included by the provider.
The complete LGPL 2.1 text is also retained in `GNU-LGPL-2.1.txt` and bundled
beside the provider notice in each wheel.
The upstream notice contains original NumPy paths; the wheel's `manifest.json`
records the corresponding rparity paths, hashes, and sizes.

The provider's [build repository](https://github.com/MacPython/openblas-libs/tree/2b6df42890977bb66d203ab1f2c85c7879a36f7a)
records its release construction. Numerical source is available from
[OpenBLAS](https://github.com/OpenMathLib/OpenBLAS/) and the
[GCC project](https://gcc.gnu.org/git/?p=gcc.git;a=tree).
These general numerical libraries contain no R package implementation.
