from __future__ import annotations

from enum import Enum

from .generated import (
    F8DataTypeSchema,
    F8PrimitiveTypeEnum,
    F8PrimitiveTypeSchema,
    F8ArrayTypeSchema,
    F8ArrayTypeKind,
    F8AnyTypeSchema,
    F8AnyTypeKind,
    F8ComplexObjectTypeSchema,
    F8ComplexTypeKind,
)

def schema_type(schema: F8DataTypeSchema) -> str:
    inner = schema.root
    value = inner.type
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def schema_default(schema: F8DataTypeSchema) -> any:
    return getattr(schema.root, "default", None)


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


def string_schema(*, default: str | None = None, enum: list[str] | None = None) -> F8PrimitiveTypeSchema:
    return F8PrimitiveTypeSchema(type=F8PrimitiveTypeEnum.string, default=default, enum=enum)


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


def array_schema(
    *,
    items: F8DataTypeSchema,
) -> F8ArrayTypeSchema:
    return F8ArrayTypeSchema(
        type=F8ArrayTypeKind.array,
        items=items,
    )


def any_schema() -> F8AnyTypeSchema:
    return F8AnyTypeSchema(
        type=F8AnyTypeKind.any,
    )


def complex_object_schema(
    *,
    properties: dict[str, F8DataTypeSchema],
) -> F8ComplexObjectTypeSchema:
    return F8ComplexObjectTypeSchema(
        type=F8ComplexTypeKind.object,
        properties=properties,
    )
