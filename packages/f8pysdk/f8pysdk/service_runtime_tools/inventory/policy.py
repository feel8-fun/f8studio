from __future__ import annotations

import os
from collections.abc import Sequence


DISABLED_SERVICE_CLASSES_ENV = "F8_DISABLED_SERVICE_CLASSES"


def split_service_class_values(values: Sequence[str]) -> tuple[str, ...]:
    service_classes: list[str] = []
    for value in values:
        for comma_part in str(value or "").split(","):
            for pathsep_part in comma_part.split(os.pathsep):
                item = pathsep_part.strip()
                if item:
                    service_classes.append(item)
    return tuple(service_classes)


def merge_disabled_service_classes(
    *,
    explicit_service_classes: Sequence[str] | None = None,
    include_env: bool = True,
) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    sources: list[Sequence[str]] = []
    if explicit_service_classes is not None:
        sources.append(explicit_service_classes)
    if include_env:
        sources.append((os.environ.get(DISABLED_SERVICE_CLASSES_ENV) or "",))

    for source in sources:
        for service_class in split_service_class_values(source):
            if service_class in seen:
                continue
            seen.add(service_class)
            merged.append(service_class)
    return tuple(merged)


__all__ = [
    "DISABLED_SERVICE_CLASSES_ENV",
    "merge_disabled_service_classes",
    "split_service_class_values",
]
