"""Generate an auditable option-level table from golden cases and test results."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def source_digest(root: Path) -> str:
    """Fingerprint the validated runtime source without exposing local paths."""
    digest = hashlib.sha256()
    for path in sorted((root / 'src' / 'rparity').rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def generate(root: Path | None = None) -> str:
    """Write docs/coverage.md; unexecuted fixtures never count as passing."""
    root = root or Path.cwd()
    validation_path = root / 'reports' / 'validation.json'
    validation: dict[str, Any] = {}
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        if validation.get('source_sha256') != source_digest(root):
            validation = {}
    outcomes = validation.get('cases', {})
    first_complete = validation.get('stage1_complete', False)
    second_complete = validation.get('stage2_complete', False)
    rows: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in sorted((root / 'tests' / 'golden').glob('*/*.json')):
        case = json.loads(path.read_text())
        module = case.get('module', path.parent.name)
        option = case.get('option', case.get('spec', {}).get('call', module))
        rows[(module, option)].append(case.get('id', path.stem))
    lines = [
        '# Validation coverage', '',
        ('Stage 1 acceptance checks are complete for v0.1.0.' if first_complete else
         'Stage 1 remains in progress until every required option and acceptance check passes.'), '',
        ('Stage 2 acceptance checks are complete for v0.2.0.' if second_complete else
         'Stage 2 verification is ongoing; generated cases alone do not establish acceptance.'), '',
        'Counts below refer to committed synthetic R observations. Passing status comes from',
        'the latest recorded pytest run; unexecuted or skipped cases are not passes.', '',
        '| Module | R function and options | Implementation | Golden cases | Pass rate | Notes |',
        '| --- | --- | --- | ---: | ---: | --- |',
    ]
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for (module, option), ids in sorted(rows.items()):
        passed = sum(outcomes.get(i, {}).get('status') == 'passed' for i in ids)
        failed = sum(outcomes.get(i, {}).get('status') == 'failed' for i in ids)
        verified = passed + failed
        rate = f'{passed / len(ids):.2%}' if verified else 'unverified'
        notes = f'{failed} failures; {len(ids)-verified} unverified'
        option = option.replace('|', '\\|')
        complete = second_complete if module in {'gam', 'tmb'} else first_complete
        implementation = 'implemented and validated' if complete else 'implemented, verification ongoing'
        lines.append(f'| {module} | {option} | {implementation} | {len(ids)} | {rate} | {notes} |')
        totals[module][0] += len(ids)
        totals[module][1] += passed
        totals[module][2] += failed
    if not rows:
        for module in ['lmer', 'glmer', 'inference', 'anova', 'emm', 'gls']:
            lines.append(f'| {module} | required Stage 1 options | implementation in progress | 0 | unverified | Oracle generation pending |')
    lines.extend(['', '## Module totals', '', '| Module | Golden cases | Passed | Failed |', '| --- | ---: | ---: | ---: |'])
    for module, values in sorted(totals.items()):
        lines.append(f'| {module} | {values[0]} | {values[1]} | {values[2]} |')
    n = sum(v[0] for v in totals.values())
    p = sum(v[1] for v in totals.values())
    lines.extend(['', f'Total: {n} synthetic cases, {p} recorded passes.', '',
                  'Stage 1 requires at least 300 cases per module, 3,000 total and 98% passing.',
                  'Stage 2 separately requires at least 2,000 cases and 98% passing, with no new Stage 1 failures.', '',
                  '## R package examples', '',
                  'Built-in R example data is not committed. The following `needs_r` cases',
                  'are included in coverage and require the development oracle.', '',
                  '| Representative example | Result |', '| --- | --- |'])
    examples = validation.get('r_examples', {})
    for name, result in sorted(examples.items()):
        label = name.replace('|', '\\|')
        lines.append(f'| {label} | {result["status"]} |')
    if not examples:
        lines.append('| Representative examples | unverified |')
    lines.append('')
    content = '\n'.join(lines)
    destination = root / 'docs' / 'coverage.md'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content)
    return content


def main() -> None:
    """Regenerate the repository coverage document."""
    print(generate())


if __name__ == '__main__':
    main()
