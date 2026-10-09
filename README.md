![Status: Stage 1 in progress (pre-alpha, API may change)](https://img.shields.io/badge/Status-Stage%201%20in%20progress-orange)

# rparity

R-grade mixed models, marginal means, GAMs and meta-analysis in pure Python, verified against R, no R required.

Stage 1 is under development. See [progress](PROGRESS.md) and [validation coverage](docs/coverage.md).

## Development

```sh
uv sync
uv run pytest
uv run ruff check
uv run mypy src
```

R is a development-only black-box oracle. It is never imported or invoked by the runtime package. This project follows the [clean-room rules](CLEANROOM.md).
