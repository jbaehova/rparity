# Clean-room record

## Stage 1 implementation

No target R package source or function body was opened, read, or searched during this stage. The Python implementation derives from the statistical papers and textbooks recorded in REFERENCES.md, with permissively licensed NumPy, SciPy, formulaic and statsmodels dependencies.

Every numerical R observation passed through `oracle/run_case.R`. The oracle calls exported fitting, inference and prediction functions with explicit contrasts, optimizer controls and degrees-of-freedom options. It extracts public coefficients, covariance matrices, likelihoods, conditional modes, model summaries, warnings and inference tables. Function bodies and internal source files are not part of this interface.

Seeded synthetic inputs and public observations form the committed golden corpus. Precision attempts are observations rather than implementation templates. R's built-in example datasets remain only in ignored `oracle/cache/`; their tests require the development R installation. Package and R versions are recorded in `oracle/versions.json`.

The release wheel contains Python runtime modules and requires no R executable or rpy2. The same public oracle supplies representative-example checks and same-machine fit-time measurements. Required-field discrepancies remain visible in the numerical coverage and Stage 1 report.
