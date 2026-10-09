# Development and validation

This directory contains maintainer tools and numerical evidence. To use the
Python library, start with the [README](../README.md), [documentation](../docs/),
or [Python guides](../docs/guides/).

| Location | Purpose |
| --- | --- |
| [`oracle/`](oracle/README.md) | Public R API driver, seeded fixture generators, and local benchmarks. |
| [`reports/`](reports/) | Recorded model comparisons, retained differences, and distribution audits. |
| [`CLEANROOM.md`](CLEANROOM.md) | Public-reference and cleanroom implementation record. |
| [`REFERENCES.md`](REFERENCES.md) | Mathematical and public API references. |
| [`DECISIONS.md`](DECISIONS.md) | Implementation decisions and numerical conventions. |
| [`PROGRESS.md`](PROGRESS.md) | Historical development log. |

The installed library does not use these tools and does not require R. Runtime
code lives in [`src/rparity/`](../src/rparity/), runnable user examples live in
[`examples/`](../examples/), and tests live in [`tests/`](../tests/).

## Local checks

Run checks from the repository root:

```sh
uv sync --locked
uv run pytest
uv run ruff check
uv run mypy src
uv run mkdocs build --strict
```

Tests run locally. The test workflow accepts manual dispatch only and requires
an explicit request. GitHub Actions deploys documentation when documentation
changes.

The tests marked `needs_r` require the development R packages recorded in
[`oracle/versions.json`](oracle/versions.json). Every R observation passes through
[`oracle/run_case.R`](oracle/run_case.R). Its [README](oracle/README.md) describes
the allowlisted requests and fixture regeneration commands. Built-in R example
data and raw local logs stay in the ignored `development/oracle/cache/` directory.

After a full local run, record sanitized outcomes and regenerate the coverage
page:

```sh
uv run pytest --junitxml=development/oracle/cache/pytest-current.xml
uv run python development/oracle/record_validation.py \
  development/oracle/cache/pytest-current.xml
uv run python -m rparity._coverage
```

Unexecuted cases do not count as passing. Known numerical differences remain
in the corpus and count as failures in the published coverage.

## Historical evidence

The filenames in `reports/` preserve the original validation batches and release
audits. They provide reproducible evidence and are not the library's user-facing
organization. Published release tags retain the repository layout that existed
at release time.

Some historical JSON records and audit commands mention the former `oracle/`
or `reports/` locations. Their recorded paths remain unchanged as provenance.
Current commands use `development/oracle/` and `development/reports/`. The local
cache moved with the tooling and was not regenerated for the relocation.
