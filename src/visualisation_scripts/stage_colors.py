"""Central sleep-stage color palette for visualisation scripts.

All plots should use the same colors for comparable sleep stages to keep
figures visually consistent across publications. Import the constants or use
``get_stage_color`` whenever you need the color for a specific stage label or
identifier.
"""
from __future__ import annotations

from numbers import Integral
from typing import Any, Dict

WAKE_COLOR = "#D62728"
NREM_COLOR = "#1F77B4"
REM_COLOR = "#2CA02C"
DEFAULT_STAGE_COLOR = "#7F7F7F"

STAGE_COLORS_BY_ID: Dict[int, str] = {
    1: WAKE_COLOR,
    2: NREM_COLOR,
    3: REM_COLOR,
}

STAGE_COLORS_BY_NAME: Dict[str, str] = {
    "Wake": WAKE_COLOR,
    "NREM": NREM_COLOR,
    "REM": REM_COLOR,
}

_ALIAS_LOOKUP: Dict[Any, str] = {
    **STAGE_COLORS_BY_ID,
    **{key.lower(): value for key, value in STAGE_COLORS_BY_NAME.items()},
}

_WAKE_ALIASES = ("awake", "wake", "wakestate", "wakepercent")
_NREM_ALIASES = ("nrem", "nonrem", "nonrapideyenmovement", "nonrempercent")
_REM_ALIASES = ("rem", "rempercent")

for alias in _WAKE_ALIASES:
    _ALIAS_LOOKUP[alias] = WAKE_COLOR

for alias in _NREM_ALIASES:
    _ALIAS_LOOKUP[alias] = NREM_COLOR

for alias in _REM_ALIASES:
    _ALIAS_LOOKUP[alias] = REM_COLOR


def _normalize_stage_key(stage: Any) -> Any:
    if isinstance(stage, Integral):
        return int(stage)

    if isinstance(stage, str):
        stripped = stage.strip()
        if not stripped:
            return stripped
        alnum = "".join(char for char in stripped.lower() if char.isalnum())
        if not alnum:
            return stripped.lower()
        if alnum.isdigit():
            return int(alnum)
        return alnum

    return stage


def get_stage_color(stage: Any, default: str = DEFAULT_STAGE_COLOR) -> str:
    """Return the canonical color for a given sleep stage identifier."""

    key = _normalize_stage_key(stage)
    return _ALIAS_LOOKUP.get(key, default)
