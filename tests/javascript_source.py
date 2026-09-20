"""Select a top-level declaration for isolated JavaScript behavior tests."""


def slice_function(source: str, signature: str) -> str:
    """App declarations end with a column-zero brace, including default params."""
    start = source.index(signature)
    end = source.index("\n}", start)
    return source[start : end + 2]
