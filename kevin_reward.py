#!/usr/bin/env python
"""Deprecation shim – use `open_kevin.rewards` instead.

This file re-exports public symbols from `open_kevin.rewards` so that
legacy import paths continue to function.
"""

from warnings import warn
from open_kevin.rewards import *  # noqa: F401,F403

warn(
    "`kevin_reward` has moved to `open_kevin.rewards`; please update your imports.",
    DeprecationWarning,
    stacklevel=2,
) 