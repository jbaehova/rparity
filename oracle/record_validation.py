"""Record sanitized case outcomes from a pytest JUnit report.

Raw reports are local only: failure tracebacks can contain personal paths.
The public JSON includes case IDs, status, and failure types instead.
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from rparity._coverage import source_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('junit', type=Path)
    parser.add_argument('--output', type=Path, default=Path('reports/validation.json'))
    parser.add_argument('--stage1-complete', action='store_true',
                        help='Verify the numerical and representative-example completion gates.')
    parser.add_argument('--stage2-complete', action='store_true',
                        help='Verify Stage 2 independently and reject new Stage 1 failures.')
    args = parser.parse_args()
    ids = {}
    for file in Path('tests/golden').glob('*/*.json'):
        case = json.loads(file.read_text())
        ids[case.get('id', file.stem)] = case.get('module', file.parent.name)
    xml = ET.parse(args.junit)
    outcomes: dict[str, dict[str, object]] = {}
    examples: dict[str, dict[str, str]] = {}
    all_counts = {'passed': 0, 'failed': 0, 'skipped': 0, 'expected_failures': 0}
    for test in xml.iter('testcase'):
        failed = test.find('failure') is not None or test.find('error') is not None
        skip = test.find('skipped')
        expected_failure = skip is not None and skip.get('type') == 'pytest.xfail'
        status = 'failed' if failed or expected_failure else ('skipped' if skip is not None else 'passed')
        all_counts['expected_failures' if expected_failure else status] += 1
        name = test.get('name', '')
        if any(module in test.get('classname', '')
               for module in ('test_r_examples', 'test_stage2_r_examples')):
            examples[name] = {'status': status}
        candidate = name.partition('[')[2].rstrip(']')
        # Downstream tests reuse corpus IDs for separate inference checks.
        # Only a golden comparison owns its corpus outcome and objective
        # properties. Exact matching also prevents glmer/lmer suffix overlap.
        is_golden = 'golden' in name.partition('[')[0]
        matching = [candidate] if is_golden and candidate in ids else []
        for case_id in matching:
            current = outcomes.get(case_id, {})
            if current.get('status') == 'failed':
                continue
            entry: dict[str, object] = {'status': status, 'module': ids[case_id]}
            if expected_failure:
                entry['expected_failure'] = True
                entry['reason'] = skip.get('message', '') if skip is not None else ''
            if failed:
                node = test.find('failure')
                if node is None:
                    node = test.find('error')
                entry['failure_type'] = node.get('type', 'failure') if node is not None else 'failure'
            for prop in test.findall('properties/property'):
                if prop.get('name') == 'better_optimum':
                    entry['better_optimum'] = prop.get('value', '').lower() == 'true'
            outcomes[case_id] = entry
    report = {'command': f'uv run pytest --junitxml={args.junit.as_posix()}',
              'source_sha256': source_digest(Path.cwd()),
              'test_counts': all_counts, 'cases': outcomes, 'r_examples': examples}
    if args.stage1_complete or args.stage2_complete:
        required = {'lmer', 'glmer', 'inference', 'anova', 'emm', 'gls'}
        first_stage = {key: entry for key, entry in outcomes.items()
                       if entry['module'] in required}
        module_counts = {module: sum(entry['module'] == module for entry in first_stage.values())
                         for module in required}
        assert all(count >= 300 for count in module_counts.values()), module_counts
        assert len(outcomes) == len(ids) and len(first_stage) >= 3000
        assert all(entry['status'] in {'passed', 'failed'} for entry in outcomes.values())
        passed = sum(entry['status'] == 'passed' for entry in first_stage.values())
        assert passed / len(first_stage) >= 0.98, (passed, len(first_stage))
        assert all_counts['failed'] == 0 and all_counts['skipped'] == 0, all_counts
        assert len(examples) >= 14 and all(entry['status'] == 'passed'
                                          for entry in examples.values()), examples
        report['stage1_complete'] = True
        report['stage1_counts'] = {'cases': len(first_stage), 'passed': passed,
                                   'failed': len(first_stage) - passed}
        if args.stage2_complete:
            baseline = json.loads(Path('reports/STAGE_1_REGRESSION_BASELINE.json').read_text())
            failures = {key for key, entry in first_stage.items() if entry['status'] == 'failed'}
            new_failures = sorted(failures - set(baseline['failed_ids']))
            assert len(first_stage) == baseline['cases'], (len(first_stage), baseline['cases'])
            assert not new_failures, {'new_stage1_failures': new_failures}
            second_stage = {key: entry for key, entry in outcomes.items()
                            if entry['module'] in {'gam', 'tmb'}}
            second_passed = sum(entry['status'] == 'passed' for entry in second_stage.values())
            assert len(second_stage) >= 2000, len(second_stage)
            assert len(examples) >= 16, examples
            assert second_passed / len(second_stage) >= 0.98, (second_passed, len(second_stage))
            assert {'gam', 'tmb'} <= {entry['module'] for entry in second_stage.values()}
            report['stage2_complete'] = True
            report['stage2_counts'] = {'cases': len(second_stage), 'passed': second_passed,
                                       'failed': len(second_stage) - second_passed}
            report['stage1_regressions'] = new_failures
            report['stage1_resolved_ids'] = sorted(set(baseline['failed_ids']) - failures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'test_counts': all_counts, 'recorded_cases': len(outcomes)}))


if __name__ == '__main__':
    main()
