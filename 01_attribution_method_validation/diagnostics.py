"""
Attribution method validation — entry point
===========================================
Delegates to the submodule that matches the requested subcommand.
Each submodule can also be run standalone.

Subcommands
-----------
  stability         5% ZWD noise stability check          → stability.py
  randomization     Cascading parameter randomization     → randomization.py
  completeness      IG completeness axiom                  → completeness.py
  rise_convergence  RISE mask-count Monte-Carlo curves     → rise_convergence.py

Usage
-----
  python 01_attribution_method_validation/diagnostics.py stability
  python 01_attribution_method_validation/diagnostics.py randomization
  python 01_attribution_method_validation/diagnostics.py completeness
  python 01_attribution_method_validation/diagnostics.py rise_convergence
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

# Ensure this directory is on the path so submodules can import common.py
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attribution method validation")
    sub = p.add_subparsers(dest="subcommand", required=True)

    s = sub.add_parser("stability", help="Noise stability (3 seeds, 5-pct ZWD noise)")
    s.add_argument("--seeds",    type=int, nargs="+", default=[11, 22, 33])
    s.add_argument("--ig-steps", type=int, default=32)
    s.add_argument("--methods",  type=str, nargs="+", default=["saliency", "ig"],
                   choices=["saliency", "ig"])

    r = sub.add_parser("randomization", help="Cascading parameter randomization")
    r.add_argument("--ig-steps",  type=int, default=32)

    c = sub.add_parser("completeness", help="IG completeness axiom")
    c.add_argument("--steps", type=int, nargs="+", default=[8, 16, 32, 64])

    rc = sub.add_parser("rise_convergence",
                        help="RISE mask-count Monte-Carlo convergence")
    rc.add_argument("--checkpoints", type=int, nargs="+",
                    default=[32, 64, 128, 256, 512, 1024])

    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] "
          f"Attribution method validation: {args.subcommand}")

    if args.subcommand == "stability":
        from stability import run
    elif args.subcommand == "randomization":
        from randomization import run
    elif args.subcommand == "completeness":
        from completeness import run
    elif args.subcommand == "rise_convergence":
        from rise_convergence import run
    run(args)

    elapsed = time.time() - t_start
    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] "
          f"Done in {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
