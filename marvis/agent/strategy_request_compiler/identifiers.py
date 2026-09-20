"""Exact artifact identifier tokens shared by strategy grammar families."""

from __future__ import annotations

import re

_CANDIDATE_STABILITY_ASSET_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])candidate-asset-[0-9a-f]{32}(?![A-Za-z0-9_-])"
)

_CANDIDATE_STABILITY_POOL_ENTRY_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])pool-entry-[0-9a-f]{32}(?![A-Za-z0-9_-])"
)

_CROSS_MATRIX_CELL_SELECTION_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])cross-matrix-cell-selection-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)

_SCORECARD_CUTOFF_SELECTION_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])scorecard-cutoff-selection-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)

_AUTOMATIC_TREE_LEAF_SELECTION_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])automatic-tree-leaf-selection-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)

_INTERACTIVE_TREE_FRONTIER_SELECTION_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])interactive-tree-frontier-selection-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)

_INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"interactive-tree-frontier-group-selection-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)

_AUTOMATIC_TREE_ASSET_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])candidate-asset-[0-9a-f]{32}(?![A-Za-z0-9_-])"
)
