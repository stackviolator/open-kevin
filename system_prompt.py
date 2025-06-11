#!/usr/bin/env python
"""Compatibility shim – the authoritative `system_prompt` lives in
`open_kevin.prompts.system`.

This file will be removed in a future release.
"""

from open_kevin.prompts.system import system_prompt  # re-export for legacy imports

__all__ = ["system_prompt"] 