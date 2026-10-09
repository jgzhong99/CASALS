"""Command-line entry point for CASALS processing and research tools."""

from __future__ import annotations

import argparse
from importlib import import_module
import sys


COMMANDS = {
    "refh-export": "casals_l1b.refh_export",
    "refh-filter": "casals_l1b.refh_filter",
    "peaks": "casals_l1b.peaks_cli",
    "refh-dsm": "casals_l1b.refh_dsm",
    "refh-ground": "casals_l1b.refh_ground",
    "classify-refh": "casals_l1b.classification_cli",
    "diagnose-3dep": "research.reference.diagnose_3dep_offsets",
    "transfer-3dep": "research.reference.transfer_3dep_labels_to_casals",
    "transfer-features": "research.reference.extract_transfer_laz_local_features",
    "refh-error-summary": "research.refh.summarize_refh_error_distributions",
    "download-3dep": "tools.download_3dep_lpc",
    "detect-utm": "tools.detect_h5_utm_zone",
    "animate-pushbroom": "tools.animate_pushbroom",
    "animate-waveforms": "tools.animate_rx_waveforms",
    "view-refh": "tools.view_refh_points",
    "view-lpc-open3d": "tools.view_lpc_open3d",
    "view-lpc-qt": "tools.view_lpc_qt_classes",
    "debug-classification": "research.classification.debug_classifier_params",
    "classification-rules": "research.classification.explore_3dep_like_rules",
    "geolocation-beam-rules": "research.geolocation.geolocate_sweep_bins_from_beam_angle_rules",
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="casals",
        description="CASALS L1B processing, reference comparison, and research commands.",
    )
    parser.add_argument("command", nargs="?", choices=sorted(COMMANDS))
    if not args or args[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command = args.pop(0)
    if command not in COMMANDS:
        parser.error(f"unknown command {command!r}")
    module = import_module(COMMANDS[command])
    result = module.main(args)
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
