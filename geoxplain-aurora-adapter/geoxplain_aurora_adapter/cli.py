"""Unified command line entry point for geoxplain_aurora_adapter."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .serving.config import resolve_config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geoxplain-aurora-adapter",
        description="Configure or run the GeoXplain Aurora Adapter.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("setup", "listen"),
        help="'setup' prints install guidance and writes config; 'listen' starts the listener.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    if not raw_args:
        if not resolve_config_path().exists():
            from .serving.setup import main as setup_main

            setup_main([])
            return
        parser.print_help()
        print(f"\nConfig exists: {resolve_config_path()}")
        print("Run `geoxplain-aurora-adapter setup` to review it, or `geoxplain-aurora-adapter listen` to start a listener.")
        return

    if raw_args[0] in {"-h", "--help"}:
        parser.parse_args(raw_args)
        return

    if raw_args[0].startswith("-"):
        from .serving.setup import main as setup_main

        setup_main(raw_args)
        return

    args = parser.parse_args(raw_args[:1])
    sub_args = raw_args[1:]

    if args.command == "setup":
        from .serving.setup import main as setup_main

        setup_main(sub_args)
        return

    if args.command == "listen":
        from .remote.cli import main as listen_main

        listen_main(sub_args)
        return

    parser.error(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
