"""Console script entry point for mecanum-sim CLI."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    # Forward to scripts/run_sim.py
    root_dir = Path(__file__).resolve().parent.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from scripts.run_sim import main as run_main
    run_main()


if __name__ == "__main__":
    main()

