"""CLI for SGCC data preparation. This module never trains a classifier."""
from __future__ import annotations

import argparse
import logging

from .config import load_config
from .dataset_builders import MODELS, prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="YAML paths resolve relative to this file")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Prepare all four variants")
    selection.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        report = prepare(load_config(args.config), MODELS if args.all else [args.model])
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Preprocessing failed: {exc}\n")
    logging.info("Preparation complete: %s consumers, %s features, all artifact checks passed",
                 report["raw"]["unique_consumers"], report["features"]["count"])


if __name__ == "__main__":
    main()
