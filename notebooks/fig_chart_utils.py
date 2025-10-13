"""Chart-specific utilities for GenX figure creation."""
from __future__ import annotations

from typing import Dict

import altair as alt

from fig_config import COLOR_MAP


def title_case(value: str) -> str:
    """Convert snake_case identifiers into title case strings."""

    if isinstance(value, str):
        return value.replace("_", " ").title()
    return value


VAR_ABBR_MAP: Dict[str, str] = {
    "model": "m",
    "case": "c",
    "planning_year": "y",
    "resource_name": "rn",
    "agg_zone": "az",
    "zone": "z",
    "tech_type": "tt",
    "value": "v",
    "end_value": "ev",
    "line_name": "ln",
    "Region": "r",
}

VAR_ABBR_TITLE_MAP: Dict[str, str] = {
    code: title_case(name) for name, code in VAR_ABBR_MAP.items()
}


def configure_full_label_display(chart: alt.Chart) -> alt.Chart:
    """Ensure long labels (e.g., model names) render completely across chart elements."""

    return (
        chart.configure_axis(labelLimit=0)
        .configure_legend(labelLimit=0)
        .configure_header(labelLimit=0)
    )


def tech_color_encoding(field: str = "tt", title: str | None = None) -> alt.Color:
    """Return a consistent technology color encoding for Altair charts."""

    color_title = title or title_case("tech_type")
    return (
        alt.Color(field)
        .scale(domain=list(COLOR_MAP.keys()), range=list(COLOR_MAP.values()))
        .title(color_title)
    )
