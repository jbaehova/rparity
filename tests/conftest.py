"""Explicit known parity failures remain failures in the numerical report.

Strict xfail keeps unexpected regressions and unexpectedly fixed cases visible.
The documented corpus gate counts these cases as failures, never passes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    failures_path = Path(__file__).parent / 'golden' / 'known_failures.json'
    if not failures_path.exists():
        return
    failures = json.loads(failures_path.read_text())
    for item in items:
        if 'golden' not in item.name:
            continue
        case_id = item.name.partition('[')[2].rstrip(']')
        if case_id in failures:
            item.add_marker(pytest.mark.xfail(reason=failures[case_id]['reason'], strict=True,
                                             raises=AssertionError))
