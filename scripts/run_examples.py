#!/usr/bin/env python3
"""
Run Bluemira example files in sequence. Exit with code 1 if any errors.

This will recursively search for python files within a given directory,
and run them in sequence. It will report which ran without error, and
which failed.

By default, this script will disable plot/CAD display windows. The idea
being that this script will mostly be automatically run in a
non-interactive shell.
"""

import os
import re
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from unittest import mock

import matplotlib as mpl
import matplotlib.pyplot as plt

from bluemira.base.file import get_bluemira_path

BANNER = "\n" + "-" * 72 + "\n| {}\n" + "-" * 72 + "\n"
EXAMPLES_ROOT = os.path.normpath(get_bluemira_path("examples", subfolder=""))
EXCLUDE_PATTERNS = ["convert_py_to_ipynb"]


@dataclass
class Args:
    """Command line arguments."""

    examples_dir: str
    exclude_pattern: List[str]
    plotting_on: bool


def parse_args(sys_args: List[str]) -> Args:
    """Parse command line arguments"""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples-dir",
        default=EXAMPLES_ROOT,
        help=f"the directory in which to look for examples (default: {EXAMPLES_ROOT})",
    )
    parser.add_argument(
        "-e",
        "--exclude-pattern",
        action="append",
        default=EXCLUDE_PATTERNS,
        help=(
            "do not run example files that contain this regex pattern. "
            "This argument may be used more than once"
        ),
    )
    parser.add_argument(
        "--plotting-on",
        action="store_true",
        default=False,
        help="enable interactive plotting windows",
    )
    args = parser.parse_args(sys_args)
    return Args(**vars(args))


def find_python_files(examples_dir: str, exclude_patterns: List[str]) -> List[str]:
    """Glob for Python files in the given directory."""
    return sorted(
        [
            path
            for path in Path(examples_dir).rglob("*.py")
            if not any(re.search(p, str(path)) for p in exclude_patterns)
        ]
    )


def run_example(file_path: str) -> bool:
    """Run the given Python file; return True if no errors, else False."""
    source = Path(file_path).read_text()
    try:
        exec(compile(source, file_path, "exec"), globals())  # noqa: S102
    except Exception as e:  # noqa: BLE001
        print(e, file=sys.stderr)
        return False
    finally:
        plt.close("all")
    return True


def run_examples(
    example_files: List[str], plotting_on: bool = False
) -> List[Tuple[str, bool]]:
    """
    Run the given example files.

    Returns the list of examples run, along with a boolean indicating
    whether there were any errors when running the example (True
    indicating no errors, and False the opposite).
    """
    if not plotting_on:
        mpl.use("Agg")
        # Disable CAD viewer by mocking out FreeCAD API's displayer.
        # Note that if we use a new CAD backend, this must be changed.
        mock.patch("bluemira.codes._freecadapi.show_cad").start()

    failed = []
    for example in example_files:
        display_path = Path(example).relative_to(args.examples_dir)
        print(BANNER.format(display_path))
        result = run_example(example)
        # Flush stdout so we don't get un-flushed output from previous
        # examples under this example's banner
        sys.stdout.flush()
        failed.append((display_path, result))
    return failed


if __name__ == "__main__":
    # Make external_code example run
    sys.path.insert(0, str(Path(Path(__file__).parent.parent, "examples", "codes")))

    args = parse_args(sys.argv[1:])
    example_py_files = find_python_files(args.examples_dir, args.exclude_pattern)
    if not example_py_files:
        print(f"found no python files in '{args.examples_dir}'.")
        sys.exit(1)

    results = run_examples(example_py_files, args.plotting_on)
    print("\nExamples run:")
    print("-------------")
    for example, result in results:
        print(f"{'✔️' if result else '❌'} {example}")
    sys.exit(not all(r[1] for r in results))
