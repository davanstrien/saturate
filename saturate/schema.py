"""Type rules for ParquetSink: what a pinned column may hold and the type its values are stored
as (dynamic mode), and which record keys a declared schema lacks (declared mode).

ParquetSink owns the pin (the schema the first parts fixed) and calls these pure functions to
check each row against it. Nothing here does I/O or holds state.
"""

from __future__ import annotations

import pyarrow as pa


def is_list(arrow_type: pa.DataType) -> bool:
    return pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type)


def is_nested(arrow_type: pa.DataType) -> bool:
    return pa.types.is_struct(arrow_type) or is_list(arrow_type)


def is_untyped(arrow_type: pa.DataType) -> bool:
    """A type with no typed leaf (null, list<null>, struct<>, ...): inferred from values that
    carry no information, so it can never be a pin — a later real value must be free to set it."""
    if pa.types.is_null(arrow_type):
        return True
    if is_list(arrow_type):
        return is_untyped(arrow_type.value_type)
    if pa.types.is_struct(arrow_type):
        return all(is_untyped(field.type) for field in arrow_type)
    return False


def without_untyped(arrow_type: pa.DataType) -> pa.DataType | None:
    """`arrow_type` with every untyped field removed, or None when nothing typed is left: a
    struct field only ever seen as None (or as empty lists) is dropped, and so is a struct left
    with no field. This is the type a parquet file stores and the pin holds. A null type is never
    written, because a later file that types the field cannot be read beside it
    (pq.read_table(dir) cannot cast string to null)."""
    if pa.types.is_null(arrow_type):
        return None
    if is_list(arrow_type):
        item = without_untyped(arrow_type.value_type)
        if item is None:
            return None
        return pa.large_list(item) if pa.types.is_large_list(arrow_type) else pa.list_(item)
    if pa.types.is_struct(arrow_type):
        fields = []
        for field in arrow_type:
            typed = without_untyped(field.type)
            if typed is not None:
                fields.append(pa.field(field.name, typed, field.nullable))
        return pa.struct(fields) if fields else None
    return arrow_type


def without_untyped_schema(schema: pa.Schema) -> pa.Schema:
    """`schema` with its untyped columns dropped and the untyped fields removed from the rest
    (`without_untyped`)."""
    fields = []
    for field in schema:
        typed = without_untyped(field.type)
        if typed is not None:
            fields.append(pa.field(field.name, typed, field.nullable))
    return pa.schema(fields)


def _has_empty_nested_struct(arrow_type: pa.DataType, top: bool = True) -> bool:
    """Parquet cannot write a struct with no fields below the top level (a nested `{}`); a
    top-level one is an untyped column and is simply dropped."""
    if pa.types.is_struct(arrow_type):
        if not top and len(arrow_type) == 0:
            return True
        return any(_has_empty_nested_struct(field.type, False) for field in arrow_type)
    if is_list(arrow_type):
        return _has_empty_nested_struct(arrow_type.value_type, False)
    return False


def _has_repeated_field_name(arrow_type: pa.DataType) -> bool:
    """A struct with two fields of one name, at any depth. A dict cannot make one, but an Arrow
    StructScalar in a record can; a cast would keep only one of the two values."""
    if pa.types.is_struct(arrow_type):
        names = [field.name for field in arrow_type]
        if len(names) != len(set(names)):
            return True
        return any(_has_repeated_field_name(field.type) for field in arrow_type)
    if is_list(arrow_type):
        return _has_repeated_field_name(arrow_type.value_type)
    return False


def check_storable(column: str, arrow_type: pa.DataType) -> None:
    """Raise TypeError when Parquet cannot store values of `arrow_type` as they are: a nested
    empty object, or a struct with a repeated field name."""
    if _has_empty_nested_struct(arrow_type):
        raise TypeError(f"{column}: {arrow_type} has a nested empty object, which parquet cannot store")
    if _has_repeated_field_name(arrow_type):
        raise TypeError(f"{column}: {arrow_type} has a struct with a repeated field name")


def _is_number(arrow_type: pa.DataType) -> bool:
    return pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type)  # bool is not an integer here


def _is_text(arrow_type: pa.DataType) -> bool:
    return pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type)


def _is_bytes(arrow_type: pa.DataType) -> bool:
    return pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type)


SCALAR_KINDS = (
    pa.types.is_decimal,
    pa.types.is_timestamp,
    pa.types.is_date,
    pa.types.is_time,
    pa.types.is_duration,
    pa.types.is_boolean,
    _is_text,
    _is_bytes,
)


def same_kind(value_type: pa.DataType, pinned_type: pa.DataType) -> bool:
    """Whether scalar values of `value_type` may be stored in a column pinned to `pinned_type`:
    the same kind of thing, with the value-level check left to a safe cast (2 fits int32; 0.5
    into int64 is refused by the cast). Kinds never cross: a float into an int column, an int
    into a string column, a bool into an int column are conflicts, because a cast there
    rewrites the value rather than records it."""
    if _is_number(value_type) and _is_number(pinned_type):
        return not (pa.types.is_floating(value_type) and pa.types.is_integer(pinned_type))
    return any(kind(value_type) and kind(pinned_type) for kind in SCALAR_KINDS)


def fit(value_type: pa.DataType, pinned_type: pa.DataType) -> pa.DataType | None:
    """The type that values of `value_type` are stored as in a column pinned to `pinned_type`;
    None when they do not fit. Usually that is `pinned_type` itself: the same type, an untyped
    one (nulls), or a scalar of the same kind (`same_kind`). Lists and structs are checked item
    by item and field by field (`_fit_struct`). A pinned field's type never changes."""
    if value_type == pinned_type or is_untyped(value_type):
        return pinned_type
    if pa.types.is_struct(value_type) and pa.types.is_struct(pinned_type):
        return _fit_struct(value_type, pinned_type)
    if is_list(value_type) and is_list(pinned_type):
        item = fit(value_type.value_type, pinned_type.value_type)
        if item is None:
            return None
        return pa.large_list(item) if pa.types.is_large_list(pinned_type) else pa.list_(item)
    return pinned_type if same_kind(value_type, pinned_type) else None


def _fit_struct(value_type: pa.StructType, pinned_type: pa.StructType) -> pa.StructType | None:
    """A struct value against a pinned struct. A pinned field the value lacks is stored as null.
    A field the pin also has must fit the pinned field's type. A typed field the pin lacks is
    appended to the pinned struct, in the order the fields arrive."""
    fields = list(pinned_type)
    pinned_names = [field.name for field in pinned_type]
    for field in value_type:
        if field.name in pinned_names:
            i = pinned_names.index(field.name)
            inner = fit(field.type, fields[i].type)
            if inner is None:
                return None
            fields[i] = pa.field(field.name, inner, fields[i].nullable)
        else:
            typed = without_untyped(field.type)
            if typed is not None:
                fields.append(pa.field(field.name, typed))
    return pa.struct(fields)


def cast_columns(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """`table` with each column cast to its type in `schema`. The cast is safe (values that do
    not fit raise) and matches struct fields by name."""
    for i, field in enumerate(table.schema):
        target = schema.field(field.name).type
        if target != field.type:
            table = table.set_column(i, pa.field(field.name, target), table.column(i).cast(target))
    return table


def _undeclared_keys(value, declared_type: pa.DataType, path: str) -> list[str]:
    """The paths of the dict keys in `value` (at any depth, inside lists too) that
    `declared_type` has no struct field for, e.g. "m.b" or "items[].z"."""
    found = []
    if isinstance(value, dict) and pa.types.is_struct(declared_type):
        names = [field.name for field in declared_type]
        for key, inner in value.items():
            if key in names:
                inner_type = declared_type.field(names.index(key)).type
                found.extend(_undeclared_keys(inner, inner_type, f"{path}.{key}"))
            else:
                found.append(f"{path}.{key}")
    elif isinstance(value, (list, tuple)) and is_list(declared_type):
        for item in value:
            found.extend(_undeclared_keys(item, declared_type.value_type, f"{path}[]"))
    return found


def outside_declared(record: dict, schema: pa.Schema) -> list[str]:
    """The fields of `record`, top level or nested, that the declared `schema` lacks. Arrow
    would drop an undeclared struct key without a word, losing the value."""
    found = []
    for key, value in record.items():
        if key in schema.names:
            found.extend(_undeclared_keys(value, schema.field(key).type, key))
        else:
            found.append(key)
    return sorted(set(found))
