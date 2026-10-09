# Contributing to rparity

Start with the [user documentation](https://jbaehova.github.io/rparity/)
and [runnable examples](examples/README.md) when reproducing a problem.
An issue is most useful with a small synthetic DataFrame, the model call,
expected behavior and Python/package versions. Avoid sharing sensitive data.

## Local development

```sh
git clone https://github.com/jbaehova/rparity.git
cd rparity
uv sync
uv run pytest
uv run ruff check
uv run mypy src
uv run mkdocs build --strict
```

Tests run locally. The GitHub Actions test workflow is dispatched only on
explicit request; documentation changes deploy the static site.
Tests marked `needs_r` require R and the development oracle packages.

## Where changes belong

- `src/rparity/`: implementation and public APIs.
- `examples/`: small, standalone analyses using synthetic data.
- `docs/`: installation, analysis guides, API usage and R migration.
- `tests/`: behavior and numerical regression checks.
- `development/`: R observation tools, references and validation history.

Keep public APIs typed and documented. Run the checks relevant to a change
and preserve recorded numerical tolerances and unresolved cases.
Do not use target R package implementation source when contributing code;
read the [clean-room policy](development/CLEANROOM.md) and
[mathematical references](development/REFERENCES.md).

See [development/README.md](development/README.md) for oracle regeneration
and numerical validation commands. R remains a development-only dependency.
