from __future__ import annotations

from .generated import (
    F8DataTypeSchema,
    F8PrimitiveTypeEnum,
    F8PrimitiveTypeSchema,
    F8ArrayTypeSchema,
    F8AnyTypeSchema,
    F8ComplexObjectTypeSchema,
)

OPERATOR_KEY_SEP = ":"


def operator_key(service_class: str, operator_class: str) -> str:
    """
    Build a globally-unique operator key from (serviceClass, operatorClass).

    This is the canonical identifier used by registries and editors.
    """
    service_class = str(service_class or "").strip()
    operator_class = str(operator_class or "").strip()
    if not service_class:
        raise ValueError("service_class must be non-empty")
    if not operator_class:
        raise ValueError("operator_class must be non-empty")
    return f"{service_class}{OPERATOR_KEY_SEP}{operator_class}"


def split_operator_key(key: str) -> tuple[str, str]:
    """
    Split an operator key back into (serviceClass, operatorClass).
    """
    k = str(key or "")
    if OPERATOR_KEY_SEP not in k:
        raise ValueError(f"Invalid operator key (missing '{OPERATOR_KEY_SEP}'): {k}")
    service_class, operator_class = k.split(OPERATOR_KEY_SEP, 1)
    service_class = service_class.strip()
    operator_class = operator_class.strip()
    if not service_class or not operator_class:
        raise ValueError(f"Invalid operator key: {k}")
    return service_class, operator_class


def schema_type(schema: F8DataTypeSchema) -> str:
    return schema.type


def schema_default(schema: F8DataTypeSchema) -> any:
    return schema.default


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
    items: F8PrimitiveTypeSchema,
) -> F8ArrayTypeSchema:
    return F8ArrayTypeSchema(
        type="array",
        items=items,
    )


def any_schema() -> F8AnyTypeSchema:
    return F8AnyTypeSchema(
        type="any",
    )


def complex_object_schema(
    *,
    properties: dict[str, F8DataTypeSchema],
) -> F8ComplexObjectTypeSchema:
    return F8ComplexObjectTypeSchema(
        type="object",
        properties=properties,
    )
