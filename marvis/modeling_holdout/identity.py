"""Private bounded row identity, stable across Parquet encoding and row order.

Only custody/admission code uses this decoder. Neither rows nor member hashes
are returned to a developer, Agent or evaluator before model freeze.
"""
from datetime import date, datetime, time
from decimal import Decimal, Context
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from contextlib import contextmanager
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pcsv

from marvis.modeling_holdout.contracts import HoldoutError

MAX_SEALED_BYTES = 64 * 1024 * 1024
MAX_DECODED_BYTES = 128 * 1024 * 1024
MAX_SEALED_ROWS = 1_000_000
MAX_SEALED_COLUMNS = 128


def bounded_parquet(path, *, columns=None):
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    if columns is None:
        decoded = sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups))
        oversized = path.stat().st_size > MAX_SEALED_BYTES or len(parquet.schema_arrow) > MAX_SEALED_COLUMNS
    else:
        selected = set(columns)
        # Do not decode unrelated nested/padded columns. Their size or type must
        # not disable checking an already reserved primitive projection.
        decoded = sum(metadata.row_group(i).column(j).total_uncompressed_size
                      for i in range(metadata.num_row_groups)
                      for j in range(metadata.row_group(i).num_columns)
                      if metadata.row_group(i).column(j).path_in_schema.split('.')[0] in selected)
        oversized = len(selected) > MAX_SEALED_COLUMNS
    if oversized or metadata.num_rows > MAX_SEALED_ROWS or decoded > MAX_DECODED_BYTES:
        raise HoldoutError('holdout_resource_profile_exceeded', 413)
    return parquet


def supported_columns(schema, *, exclude_columns=('split',)):
    columns = []
    for field in schema:
        if field.name in set(exclude_columns):
            continue
        try:
            _type(field.type)
        except HoldoutError:
            continue
        columns.append(field.name)
    return sorted(columns)


@contextmanager
def identity_source(path):
    """Normalize only inside custody/admission, never into a public artifact."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {'.csv', '.xls', '.xlsx', '.xlsm'}:
        from marvis.data.excel_ingest import detect_excel_container_format, ingest_sheet, list_sheets
        excel = detect_excel_container_format(path)
        if excel is not None:
            if path.stat().st_size > MAX_SEALED_BYTES:
                raise HoldoutError('holdout_resource_profile_exceeded', 413)
            if excel != 'xls':
                with zipfile.ZipFile(path) as archive:
                    if sum(item.file_size for item in archive.infolist()) > MAX_DECODED_BYTES:
                        raise HoldoutError('holdout_resource_profile_exceeded', 413)
            with tempfile.TemporaryDirectory(prefix='marvis-private-admission-') as temporary:
                normalized, _ = ingest_sheet(path, list_sheets(path)[0], Path(temporary), max_rows=MAX_SEALED_ROWS)
                yield pq.ParquetFile(normalized), normalized
            return
    if suffix == '.csv':
        # Arrow's numeric parser preserves round-trip float serialization. The
        # identity representation also unifies textual and typed primitives.
        if path.stat().st_size > MAX_SEALED_BYTES:
            raise HoldoutError('holdout_resource_profile_exceeded', 413)
        for encoding in ('utf-8-sig', 'gb18030', 'latin-1'):
            try:
                reader = pcsv.open_csv(path, read_options=pcsv.ReadOptions(block_size=65536, encoding=encoding))
                break
            except (UnicodeError, pa.ArrowInvalid):
                continue
        else:
            raise HoldoutError('holdout_identity_schema_invalid', 422)
        try:
            yield reader, path
        finally:
            reader.close()
        return
    if suffix in {'.feather', '.ftr'}:
        if path.stat().st_size > MAX_SEALED_BYTES:
            raise HoldoutError('holdout_resource_profile_exceeded', 413)
        with pa.memory_map(str(path), 'r') as stream:
            yield pa.ipc.open_file(stream), path
        return
    yield pq.ParquetFile(path), path


def source_schema(source):
    return source.schema_arrow if isinstance(source, pq.ParquetFile) else source.schema


def source_batches(source, columns):
    if isinstance(source, pq.ParquetFile):
        yield from source.iter_batches(batch_size=2048, columns=columns)
    elif isinstance(source, pa.ipc.RecordBatchFileReader):
        for index in range(source.num_record_batches):
            yield source.get_batch(index).select(columns)
    else:
        for batch in source:
            yield batch.select(columns)


_NUMERIC_TEXT = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')


def _value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float, Decimal)) or (isinstance(value, str) and _NUMERIC_TEXT.fullmatch(value)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        number = Decimal(str(value))
        return '0' if number == 0 else str(number.normalize(context=Context(prec=max(100, len(number.as_tuple().digits)))))
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return ['bytes', value.hex()]
    if isinstance(value, str):
        return value
    raise HoldoutError('holdout_identity_type_unsupported', 422)


def _type(dtype):
    if pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_decimal(dtype):
        return 'number'
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return 'string'
    if pa.types.is_boolean(dtype):
        return 'boolean'
    if pa.types.is_date(dtype) or pa.types.is_timestamp(dtype) or pa.types.is_time(dtype):
        return 'date'
    if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
        return 'bytes'
    if pa.types.is_null(dtype):
        return 'null'
    if pa.types.is_dictionary(dtype):
        return _type(dtype.value_type)
    raise HoldoutError('holdout_identity_type_unsupported', 422)


def parquet_identity(path, *, exclude_columns=('split',), columns=None):
    with identity_source(path) as (source, normalized):
        schema = source_schema(source)
        selected = set(columns) if columns is not None else set(schema.names) - set(exclude_columns)
        fields = sorted(field.name for field in schema if field.name in selected)
        if not fields or set(fields) != selected or len(fields) > MAX_SEALED_COLUMNS or len(set(schema.names)) != len(schema.names):
            raise HoldoutError('holdout_identity_schema_invalid', 422)
        for field in schema:
            if field.name in selected:
                _type(field.type)
        if isinstance(source, pq.ParquetFile):
            bounded_parquet(normalized, columns=fields if columns is not None else None)
        # Schema types and serialization widths do not reopen a cohort. Identity
        # conservatively unifies primitive lexical representations across formats.
        header = json.dumps(fields, ensure_ascii=False, separators=(',', ':')).encode()
        rows, decoded = [], 0
        for batch in source_batches(source, fields):
            decoded += batch.nbytes
            if decoded > MAX_DECODED_BYTES or len(rows) + batch.num_rows > MAX_SEALED_ROWS:
                raise HoldoutError('holdout_resource_profile_exceeded', 413)
            for row in batch.to_pylist():
                encoded = json.dumps([_value(row[name]) for name in fields], ensure_ascii=False, separators=(',', ':')).encode()
                rows.append(hashlib.sha256(header + b'\0' + encoded).digest())
        fingerprint = hashlib.sha256(header + b'\0' + b''.join(sorted(rows))).hexdigest()
        return fingerprint, frozenset(row.hex() for row in rows)
