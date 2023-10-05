#!/usr/bin/env python3
"""
Take a '.report.json' output from pytest-json-report, and generate a
markdown formatted report of any warnings generated.

The script can also compare one JSON report to another, 'baseline',
report and generate a markdown report that lists new warnings.

In compare mode, the exit code is the number of new warnings. In normal
mode, the exit code is the number of warnings. A negative exit code
indicates an error.
"""

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set, Tuple

INDENT_SIZE = 2
PROG = "format_warning_report"
WARNINGS_KEY = "warnings"


@dataclass(frozen=True, eq=True)
class Warning:  # noqa: A001
    """Stores information from a Python warning."""

    message: str
    category: str
    when: str
    filename: str
    lineno: int

    def __str__(self) -> str:
        """Convert to formatted string."""
        return f"{self.filename}:{self.lineno}: {self.category}: {self.message}"


def parse_args(sys_args: List[str]) -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(PROG, description=__doc__)
    parser.add_argument("report_file", help="path to .report.json to parse")
    parser.add_argument(
        "--compare",
        default=None,
        help="path to .report.json to compare warnings to",
        metavar="baseline_report_file",
    )
    return parser.parse_args(sys_args)


def load_warnings(file_path: str) -> Set[Warning]:
    """Load a set of warnings from file."""
    with open(file_path) as f:
        data = json.load(f)
    try:
        return {Warning(**warning) for warning in data[WARNINGS_KEY]}
    except KeyError:  # no warnings found
        return set()


def format_warnings_list(warnings: Iterable[Warning]) -> List[str]:
    """Format the list of warnings into lines of markdown."""
    whens: Dict[str, List[Warning]] = {}
    for warning in warnings:
        try:
            whens[warning.when].append(warning)
        except KeyError:  # noqa: PERF203
            whens[warning.when] = [warning]
    lines = []
    for when, warns in whens.items():
        lines.append(f"#### On {when}\n")
        lines.extend([f"- `{warn}`" for warn in warns])
        lines.append("")
    return lines[:-1]  # we do not need a trailing new line


def make_collapsable(md_lines: List[str], summary: str) -> List[str]:
    """Surround the given lines in html to make them collapsable."""
    lines = [
        "<details>\n",
        f"{' '*INDENT_SIZE}<summary>{summary}</summary>\n",
    ]
    lines.extend(md_lines)
    lines.append("\n</details>")
    return lines


def elements_not_in(head: Iterable[Any], ref: Iterable[Any]) -> List:
    """Find the elements that are in head, but not ref."""
    return [head_el for head_el in head if head_el not in ref]


def compare_warnings(
    head_warnings: Set[Warning], ref_warnings: Set[Warning]
) -> Tuple[List[Warning], List[Warning]]:
    """Find new and fixed warnings in 'head' using 'ref' as baseline."""
    new_warnings = elements_not_in(head_warnings, ref_warnings)
    fixed_warnings = elements_not_in(ref_warnings, head_warnings)
    return new_warnings, fixed_warnings


def format_warning_report(sys_args: List[str]) -> int:
    """Run the script."""
    inputs = parse_args(sys_args)
    warnings = load_warnings(inputs.report_file)
    exit_code = len(warnings)
    report = ["### ⚠️ Warning Report\n"]
    warning_list_md = format_warnings_list(warnings)
    if inputs.compare is not None:
        ref_warnings = load_warnings(inputs.compare)
        new, fixed = compare_warnings(warnings, ref_warnings)
        tada = " 🎉" if len(new) == 0 else ""
        report.append(
            f"Found {len(new)} new warning{'' if len(new) == 1 else 's'}, "
            f"{len(fixed)} fixed warning{'' if len(fixed) == 1 else 's'}.{tada}\n"
        )
        if new:
            warning_list = format_warnings_list(new)
            report.extend(
                make_collapsable(warning_list, f"New warnings ({len(warnings)})")
            )
        exit_code = len(new)
    else:
        plural = "" if len(warnings) == 1 else "s"
        tada = " 🎉" if len(warnings) == 0 else ""
        report.append(f"Found {len(warnings)} warning{plural}.{tada}\n")
    if warnings:
        report.extend(
            make_collapsable(warning_list_md, f"All warnings ({len(warnings)})")
        )
    print("\n".join(report))
    return exit_code


if __name__ == "__main__":
    import sys

    try:
        exit_code = format_warning_report(sys.argv[1:])
    except Exception as exc:  # noqa: BLE001
        print(exc)
        exit_code = -1
    sys.exit(exit_code)
