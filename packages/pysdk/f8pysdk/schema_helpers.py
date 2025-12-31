from __future__ import annotations

from .generated import F8PrimitiveTypeEnum, F8PrimitiveTypeSchema


def number_schema(
    *,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> F8PrimitiveTypeSchema:
    return F8PrimitiveTypeSchema(
        type=F8PrimitiveTypeEnum.number,
        default=default,
        minimum=minimum,
        maximum=maximum,
    )


def string_schema(*, default: str | None = None) -> F8PrimitiveTypeSchema:
    return F8PrimitiveTypeSchema(type=F8PrimitiveTypeEnum.string, default=default)


def integer_schema(
    *,
    default: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> F8PrimitiveTypeSchema:
    return F8PrimitiveTypeSchema(
        type=F8PrimitiveTypeEnum.integer,
        default=default,
        minimum=minimum,
        maximum=maximum,
    )


def boolean_schema(*, default: bool | None = None) -> F8PrimitiveTypeSchema:
    return F8PrimitiveTypeSchema(type=F8PrimitiveTypeEnum.boolean, default=default)

