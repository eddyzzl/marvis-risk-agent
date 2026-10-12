"""Lossless private native receipts, including explicit non-finite diagnostics.

This codec never promotes a non-finite value into a valid measured result. Native
files and committed database bytes remain the evidence authority. Public run
metadata continues to use strict JSON and cannot contain these extensions.
"""
import math
import struct

NUMBER = "__marvis_native_float64_v1__"
MAPPING = "__marvis_native_mapping_v1__"
ENCODING = "marvis.lossless-native-json.v1"


def encode_native(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {NUMBER: struct.pack(">d", value).hex()}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("native private JSON mappings require string keys")
        pairs = [(key, encode_native(item)) for key, item in value.items()]
        if any(key in {NUMBER, MAPPING} for key, _ in pairs):
            return {MAPPING: [[key, item] for key, item in pairs]}
        return dict(pairs)
    if isinstance(value, list):
        return [encode_native(item) for item in value]
    return value


def decode_native(value):
    if isinstance(value, list):
        return [decode_native(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {NUMBER}:
        bits = value[NUMBER]
        if not isinstance(bits, str) or len(bits) != 16:
            raise ValueError("invalid native floating-point encoding")
        decoded = struct.unpack(">d", bytes.fromhex(bits))[0]
        if math.isfinite(decoded):
            raise ValueError("finite number must use ordinary JSON")
        return decoded
    if set(value) == {MAPPING}:
        pairs = value[MAPPING]
        if not isinstance(pairs, list) or any(
            not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str)
            for pair in pairs
        ) or len({pair[0] for pair in pairs}) != len(pairs):
            raise ValueError("invalid native mapping encoding")
        return {key: decode_native(item) for key, item in pairs}
    return {key: decode_native(item) for key, item in value.items()}
