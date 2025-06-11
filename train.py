#!/usr/bin/env python
"""Legacy entry point kept for backwards compatibility.

The original training script has been relocated to
`open_kevin.cli.train`.  This stub allows existing tooling (e.g.
accelerate launch train.py) to continue working without changes.
"""

from open_kevin.cli.train import main as _kevin_main

if __name__ == "__main__":  # pragma: no cover
    _kevin_main()