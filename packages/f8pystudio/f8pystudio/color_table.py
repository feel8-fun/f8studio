from __future__ import annotations

from typing import Iterable


# A small, vivid palette intended for distinct input channels (ports + curves).
# Keep values RGB (no alpha); UI code can append 255 when needed.
DEFAULT_SERIES_COLORS: list[tuple[int, int, int]] = [
    (244, 67, 54),  # red
    (33, 150, 243),  # blue
    (76, 175, 80),  # green
    (255, 193, 7),  # amber
    (156, 39, 176),  # purple
    (0, 188, 212),  # cyan
    (255, 87, 34),  # deep orange
    (205, 220, 57),  # lime
    (121, 85, 72),  # brown
    (63, 81, 181),  # indigo
]


def series_color(index: int) -> tuple[int, int, int]:
    """
    Deterministic palette lookup for a 0-based series index.
    """
    pal = DEFAULT_SERIES_COLORS
    if not pal:
        return (200, 200, 200)
    return pal[int(index) % len(pal)]


def series_colors(names: Iterable[str]) -> dict[str, tuple[int, int, int]]:
    """
    Deterministically assign colors for an ordered name list.
    """
    out: dict[str, tuple[int, int, int]] = {}
    for i, n in enumerate(list(names)):
        key = str(n or "").strip()
        if not key:
            continue
        out[key] = series_color(i)
    return out

