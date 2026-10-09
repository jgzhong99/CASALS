"""The small supported command-line interface for CASALS research workflows."""

from __future__ import annotations

import argparse
from importlib import import_module
import sys


GROUPS = {
    "refh": {
        "export": "casals_l1b.refh_export",
        "filter": "casals_l1b.refh_filter",
        "dsm": "casals_l1b.refh_dsm",
        "ground": "casals_l1b.refh_ground",
        "classify": "casals_l1b.classification_cli",
        "evaluate": "casals_l1b.evaluation",
    },
    "peaks": {
        "extract": "casals_l1b.peaks_cli",
        "locate": "casals_l1b.geolocation",
        "validate": "casals_l1b.geolocation",
    },
    "reference": {
        "download": "tools.download_3dep_lpc",
        "diagnose": "research.reference.diagnose_3dep_offsets",
        "transfer": "research.reference.transfer_3dep_labels_to_casals",
    },
}


def _print_help(group: str | None = None) -> None:
    if group is None:
        print("usage: casals [-h] {refh,peaks,reference} ...")
        print("\nCASALS L1B refh processing, waveform-peak geolocation, and reference comparison.")
        print("\nGroups:")
        for name, commands in GROUPS.items():
            print(f"  {name:<12} {', '.join(commands)}")
        return
    print(f"usage: casals {group} [-h] {{{','.join(GROUPS[group])}}} ...")
    print(f"\n{group} commands:")
    for command in GROUPS[group]:
        print(f"  {command}")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="casals", add_help=False)
    parser.add_argument("group")
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return 0
    group = args.pop(0)
    if group not in GROUPS:
        parser.error(f"unknown group {group!r}; choose from {', '.join(GROUPS)}")
    if not args or args[0] in {"-h", "--help"}:
        _print_help(group)
        return 0
    command = args.pop(0)
    if command not in GROUPS[group]:
        parser.error(f"unknown {group} command {command!r}; choose from {', '.join(GROUPS[group])}")
    module = import_module(GROUPS[group][command])
    if group == "peaks" and command in {"locate", "validate"}:
        args.insert(0, command)
    result = module.main(args)
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
