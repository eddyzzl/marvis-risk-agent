from dataclasses import dataclass
import ast
from hashlib import sha256
import json
from pathlib import Path
from tokenize import TokenError
from typing import Any

from IPython.core.inputtransformer2 import TransformerManager

from marvis.model_algorithms import normalize_algorithm
from marvis.notebook_io import read_notebook_bytes

REQUIRED_CONTRACT_NAMES = (
    "RMC_SAMPLE_DF",
    "RMC_SCORE_FN",
    "RMC_TARGET_COL",
    "RMC_ALGORITHM",
)
CONTRACT_SCAN_NAMES = REQUIRED_CONTRACT_NAMES + (
    "RMC_MODEL_PARAMS",
    "MODEL_HYPERPARAMETERS",
)


class NotebookContractError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        cell_index: int | None = None,
        line_number: int | None = None,
        source_excerpt: str = "",
    ) -> None:
        super().__init__(message)
        self.cell_index = cell_index
        self.line_number = line_number
        self.source_excerpt = source_excerpt
        self.notebook_path: str | None = None
        self.notebook_sha256: str | None = None


@dataclass(frozen=True)
class ContractPrecheckResult:
    missing_names: list[str]
    target_col: str | None = None
    algorithm: str | None = None
    algorithm_raw: str | None = None
    algorithm_source: str | None = None


@dataclass(frozen=True)
class _ContractBindings:
    sample_df_defined: bool = False
    score_fn_defined: bool = False
    target_col_defined: bool = False
    algorithm_defined: bool = False
    target_col: str | None = None
    algorithm: str | None = None
    algorithm_raw: str | None = None
    algorithm_source: str | None = None
    algorithm_error: str | None = None
    invalid_names: tuple[str, ...] = ()
    contract_cell_indexes: tuple[int, ...] = ()
    source_previews: tuple[str, ...] = ()


@dataclass(frozen=True)
class _AlgorithmLiteral:
    raw: str | None
    normalized: str | None
    error: str | None = None


@dataclass(frozen=True)
class RuntimeContract:
    target_col: str
    split_col: str | None
    time_col: str | None
    pmml_output_field: str
    score_decimal_places: int
    code_model_scores_path: Path
    feature_importance_path: Path | None
    model_params_path: Path | None
    sample_snapshot_path: Path | None = None
    algorithm: str = ""


def precheck_notebook_contract(notebook_or_path: Any) -> ContractPrecheckResult:
    notebook, revision = _read_notebook_snapshot(notebook_or_path)
    try:
        bindings = _contract_bindings(notebook)
        missing = _missing_contract_names(bindings)
        if missing:
            raise NotebookContractError(
                "Notebook contract check failed before execution: missing "
                + ", ".join(missing)
            )
        if bindings.invalid_names:
            raise NotebookContractError(
                "Notebook contract check failed before execution: invalid "
                + ", ".join(bindings.invalid_names)
            )
    except NotebookContractError as exc:
        _attach_notebook_revision(exc, revision)
        raise
    return ContractPrecheckResult(
        missing_names=[],
        target_col=bindings.target_col,
        algorithm=bindings.algorithm,
        algorithm_raw=bindings.algorithm_raw,
        algorithm_source=bindings.algorithm_source,
    )


def inspect_notebook_contract(notebook_or_path: Any) -> dict[str, Any]:
    """Return a bounded, read-only static summary of the notebook RMC contract."""
    notebook, revision = _read_notebook_snapshot(notebook_or_path)
    try:
        bindings = _contract_bindings(notebook)
    except NotebookContractError as exc:
        _attach_notebook_revision(exc, revision)
        raise
    missing = _missing_contract_names(bindings)
    algorithm_valid: bool | None
    if not bindings.algorithm_defined:
        algorithm_valid = None
    else:
        algorithm_valid = bindings.algorithm is not None and bindings.algorithm_error is None
    return {
        "read_only": True,
        "source": "notebook_static_scan",
        "sample_df_defined": bindings.sample_df_defined,
        "score_fn_defined": bindings.score_fn_defined,
        "target_col_defined": bindings.target_col_defined,
        "algorithm_defined": bindings.algorithm_defined,
        "target_col": bindings.target_col,
        "algorithm": bindings.algorithm,
        "algorithm_raw": bindings.algorithm_raw,
        "algorithm_source": bindings.algorithm_source,
        "algorithm_valid": algorithm_valid,
        "algorithm_error": bindings.algorithm_error,
        "missing_names": missing,
        "invalid_names": list(bindings.invalid_names),
        "contract_cell_indexes": list(bindings.contract_cell_indexes),
        "source_previews": list(bindings.source_previews),
    }


def _missing_contract_names(bindings: _ContractBindings) -> list[str]:
    missing = []
    if not bindings.sample_df_defined:
        missing.append("RMC_SAMPLE_DF")
    if not bindings.score_fn_defined:
        missing.append("RMC_SCORE_FN")
    if not bindings.target_col_defined:
        missing.append("RMC_TARGET_COL")
    if not bindings.algorithm_defined:
        missing.append("RMC_ALGORITHM")
    return missing


def _contract_bindings(notebook: Any) -> _ContractBindings:
    sample_df_defined = False
    score_fn_defined = False
    target_col_defined = False
    explicit_algorithm_defined = False
    fallback_algorithm_defined = False
    target_col: str | None = None
    algorithm: str | None = None
    algorithm_raw: str | None = None
    algorithm_source: str | None = None
    algorithm_error: str | None = None
    invalid_names: list[str] = []
    fallback_invalid_names: list[str] = []
    fallback_algorithm_raw: str | None = None
    fallback_algorithm_source: str | None = None
    fallback_algorithm_error: str | None = None
    contract_cell_indexes: list[int] = []
    source_previews: list[str] = []

    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        source = str(cell.source)
        if _mentions_required_contract_name(source):
            contract_cell_indexes.append(cell_index)
            preview = _contract_source_preview(source)
            if preview:
                source_previews.append(preview)
        tree = _parse_contract_scan_cell(source, cell_index)
        if tree is None:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "RMC_SCORE_FN":
                    score_fn_defined = True
                continue
            if isinstance(node, ast.Assign):
                target_names = _assigned_names(node.targets)
                if "RMC_SAMPLE_DF" in target_names:
                    sample_df_defined = True
                if "RMC_SCORE_FN" in target_names:
                    if _looks_callable_binding(node.value):
                        score_fn_defined = True
                    else:
                        invalid_names.append("RMC_SCORE_FN must be callable")
                if "RMC_TARGET_COL" in target_names:
                    target_col_defined = True
                    if isinstance(node.value, ast.Constant):
                        if isinstance(node.value.value, str):
                            target_col = node.value.value
                        else:
                            invalid_names.append("RMC_TARGET_COL must be a string")
                if "RMC_ALGORITHM" in target_names:
                    explicit_algorithm_defined = True
                    algorithm_literal = _algorithm_literal(
                        "RMC_ALGORITHM",
                        node.value,
                        invalid_names,
                    )
                    algorithm = algorithm_literal.normalized
                    algorithm_raw = algorithm_literal.raw
                    algorithm_source = "RMC_ALGORITHM"
                    algorithm_error = algorithm_literal.error
                if not explicit_algorithm_defined and (
                    "RMC_MODEL_PARAMS" in target_names
                    or "MODEL_HYPERPARAMETERS" in target_names
                ):
                    params_name = (
                        "RMC_MODEL_PARAMS"
                        if "RMC_MODEL_PARAMS" in target_names
                        else "MODEL_HYPERPARAMETERS"
                    )
                    invalid_count = len(fallback_invalid_names)
                    params_algorithm = _model_params_algorithm_literal(
                        node.value,
                        params_name,
                        fallback_invalid_names,
                    )
                    if params_algorithm is not None or len(fallback_invalid_names) > invalid_count:
                        fallback_algorithm_defined = True
                        if params_algorithm is not None:
                            algorithm = params_algorithm.normalized
                            fallback_algorithm_raw = params_algorithm.raw
                            fallback_algorithm_source = f"{params_name}.algorithm"
                            fallback_algorithm_error = params_algorithm.error
                continue
            if isinstance(node, ast.AnnAssign):
                target_names = _assigned_names([node.target])
                if "RMC_SAMPLE_DF" in target_names:
                    sample_df_defined = True
                if "RMC_SCORE_FN" in target_names and node.value is not None:
                    if _looks_callable_binding(node.value):
                        score_fn_defined = True
                    else:
                        invalid_names.append("RMC_SCORE_FN must be callable")
                if "RMC_TARGET_COL" in target_names:
                    target_col_defined = True
                    if isinstance(node.value, ast.Constant):
                        if isinstance(node.value.value, str):
                            target_col = node.value.value
                        else:
                            invalid_names.append("RMC_TARGET_COL must be a string")
                if "RMC_ALGORITHM" in target_names and node.value is not None:
                    explicit_algorithm_defined = True
                    algorithm_literal = _algorithm_literal(
                        "RMC_ALGORITHM",
                        node.value,
                        invalid_names,
                    )
                    algorithm = algorithm_literal.normalized
                    algorithm_raw = algorithm_literal.raw
                    algorithm_source = "RMC_ALGORITHM"
                    algorithm_error = algorithm_literal.error
                if (
                    not explicit_algorithm_defined
                    and node.value is not None
                    and (
                        "RMC_MODEL_PARAMS" in target_names
                        or "MODEL_HYPERPARAMETERS" in target_names
                    )
                ):
                    params_name = (
                        "RMC_MODEL_PARAMS"
                        if "RMC_MODEL_PARAMS" in target_names
                        else "MODEL_HYPERPARAMETERS"
                    )
                    invalid_count = len(fallback_invalid_names)
                    params_algorithm = _model_params_algorithm_literal(
                        node.value,
                        params_name,
                        fallback_invalid_names,
                    )
                    if params_algorithm is not None or len(fallback_invalid_names) > invalid_count:
                        fallback_algorithm_defined = True
                        if params_algorithm is not None:
                            algorithm = params_algorithm.normalized
                            fallback_algorithm_raw = params_algorithm.raw
                            fallback_algorithm_source = f"{params_name}.algorithm"
                            fallback_algorithm_error = params_algorithm.error

    if not explicit_algorithm_defined and fallback_invalid_names:
        invalid_names.extend(fallback_invalid_names)
    if not explicit_algorithm_defined and fallback_algorithm_defined:
        algorithm_raw = fallback_algorithm_raw
        algorithm_source = fallback_algorithm_source
        algorithm_error = fallback_algorithm_error

    return _ContractBindings(
        sample_df_defined=sample_df_defined,
        score_fn_defined=score_fn_defined,
        target_col_defined=target_col_defined,
        algorithm_defined=explicit_algorithm_defined or fallback_algorithm_defined,
        target_col=target_col,
        algorithm=algorithm,
        algorithm_raw=algorithm_raw,
        algorithm_source=algorithm_source,
        algorithm_error=algorithm_error,
        invalid_names=tuple(sorted(set(invalid_names))),
        contract_cell_indexes=tuple(dict.fromkeys(contract_cell_indexes)),
        source_previews=tuple(source_previews[:3]),
    )


def _parse_contract_scan_cell(source: str, cell_index: int) -> ast.Module | None:
    candidates = [source]
    notebook_python_source = _transform_ipython_syntax_for_static_scan(source)
    if notebook_python_source != source:
        candidates.append(notebook_python_source)

    last_error: SyntaxError | None = None
    for candidate in candidates:
        try:
            return ast.parse(candidate)
        except SyntaxError as exc:
            last_error = exc

    if _mentions_required_contract_name(source):
        assert last_error is not None
        raise NotebookContractError(
            "Notebook contract check failed before execution: "
            f"syntax error in RMC contract code cell {cell_index}, "
            f"line {last_error.lineno}: {last_error.msg}",
            cell_index=cell_index,
            line_number=last_error.lineno,
            source_excerpt=_source_excerpt(source, last_error.lineno),
        ) from last_error
    return None


def _transform_ipython_syntax_for_static_scan(source: str) -> str:
    lines = source.splitlines()
    if lines and lines[0].lstrip().startswith("%%"):
        # Scan the Python body of a cell magic while preserving source line numbers.
        lines[0] = ""
    candidate = "\n".join(lines)
    if source.endswith("\n"):
        candidate += "\n"
    try:
        return TransformerManager().transform_cell(candidate)
    except (SyntaxError, TokenError):
        return candidate


def _source_excerpt(source: str, line_number: int | None, *, radius: int = 1) -> str:
    if not line_number:
        return ""
    lines = source.splitlines()
    start = max(1, line_number - radius)
    end = min(len(lines), line_number + radius)
    excerpt = []
    for number in range(start, end + 1):
        text = lines[number - 1].expandtabs(4)
        if len(text) > 180:
            text = text[:177] + "..."
        excerpt.append(f"L{number}: {text}")
    return "\n".join(excerpt)


def _attach_notebook_revision(
    exc: NotebookContractError,
    revision: tuple[str, str] | None,
) -> None:
    if revision is None:
        return
    exc.notebook_path, exc.notebook_sha256 = revision


def _mentions_required_contract_name(source: str) -> bool:
    return any(name in source for name in CONTRACT_SCAN_NAMES)


def _assigned_names(targets: list[ast.expr]) -> set[str]:
    names: set[str] = set()
    for target in targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.update(_assigned_names(list(target.elts)))
    return names


def _looks_callable_binding(value: ast.expr) -> bool:
    if isinstance(value, ast.Lambda):
        return True
    if isinstance(value, ast.Constant):
        return False
    if isinstance(value, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
        return False
    return True


def _contract_source_preview(source: str) -> str:
    lines = []
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(name in stripped for name in CONTRACT_SCAN_NAMES):
            lines.append(stripped[:180])
        if len(lines) >= 12:
            break
    return "\n".join(lines)


def _algorithm_literal(
    name: str,
    value: ast.expr,
    invalid_names: list[str],
) -> _AlgorithmLiteral:
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        error = "must be a string literal"
        invalid_names.append(f"{name} {error}")
        return _AlgorithmLiteral(raw=None, normalized=None, error=error)
    raw = value.value
    try:
        return _AlgorithmLiteral(raw=raw, normalized=normalize_algorithm(raw))
    except ValueError as exc:
        error = str(exc)
        invalid_names.append(f"{name} value {raw!r} {error}")
        return _AlgorithmLiteral(raw=raw, normalized=None, error=error)


def _model_params_algorithm_literal(
    value: ast.expr,
    params_name: str,
    invalid_names: list[str],
) -> _AlgorithmLiteral | None:
    if not isinstance(value, ast.Dict):
        return None
    for key_node, value_node in zip(value.keys, value.values, strict=True):
        if (
            isinstance(key_node, ast.Constant)
            and key_node.value == "algorithm"
            and isinstance(value_node, ast.Constant)
        ):
            if not isinstance(value_node.value, str):
                error = "algorithm must be a string"
                invalid_names.append(f"{params_name} {error}")
                return _AlgorithmLiteral(raw=None, normalized=None, error=error)
            raw = value_node.value
            try:
                return _AlgorithmLiteral(raw=raw, normalized=normalize_algorithm(raw))
            except ValueError as exc:
                error = str(exc)
                invalid_names.append(f"{params_name} algorithm value {raw!r} {error}")
                return _AlgorithmLiteral(raw=raw, normalized=None, error=error)
    return None


def build_contract_head_cell_source(
    *,
    sample_path: Path,
    contract_meta_path: Path,
    code_scores_path: Path,
    runtime_sample_path: Path | None = None,
    feature_importance_path: Path,
    model_params_path: Path,
    package_root: Path | None = None,
) -> str:
    lines = ["# Injected by marvis v3 (head). Original notebook unchanged."]
    if package_root is not None:
        lines.extend(
            [
                "import sys as _rmc_sys",
                f"_rmc_package_root = {Path(package_root).as_posix()!r}",
                "if _rmc_package_root not in _rmc_sys.path:",
                "    _rmc_sys.path.insert(0, _rmc_package_root)",
            ]
        )
    lines.extend(
        [
            "import os as _rmc_os",
            "import pandas as _rmc_head_pd",
            "if not getattr(_rmc_head_pd.read_csv, '_rmc_encoding_fallback', False):",
            "    _rmc_original_read_csv = _rmc_head_pd.read_csv",
            "    def _rmc_read_csv_with_encoding_fallback(*args, **kwargs):",
            "        if kwargs.get('encoding') is not None:",
            "            return _rmc_original_read_csv(*args, **kwargs)",
            "        try:",
            "            return _rmc_original_read_csv(*args, **kwargs)",
            "        except UnicodeDecodeError as _rmc_utf8_error:",
            "            _rmc_source = args[0] if args else kwargs.get('filepath_or_buffer')",
            "            if not isinstance(_rmc_source, (str, bytes, _rmc_os.PathLike)):",
            "                raise",
            "            _rmc_fallback_kwargs = dict(kwargs)",
            "            _rmc_fallback_kwargs['encoding'] = 'gb18030'",
            "            try:",
            "                return _rmc_original_read_csv(*args, **_rmc_fallback_kwargs)",
            "            except UnicodeDecodeError:",
            "                raise _rmc_utf8_error",
            "    _rmc_read_csv_with_encoding_fallback._rmc_encoding_fallback = True",
            "    _rmc_head_pd.read_csv = _rmc_read_csv_with_encoding_fallback",
            f"RMC_SAMPLE_PATH = {Path(sample_path).as_posix()!r}",
            f"RMC_CONTRACT_META_PATH = {Path(contract_meta_path).as_posix()!r}",
            f"RMC_CODE_SCORES_PATH = {Path(code_scores_path).as_posix()!r}",
            f"RMC_RUNTIME_SAMPLE_PATH = {Path(runtime_sample_path or code_scores_path.with_name('runtime_sample.csv')).as_posix()!r}",
            f"RMC_FEATURE_IMPORTANCE_PATH = {Path(feature_importance_path).as_posix()!r}",
            f"RMC_MODEL_PARAMS_PATH = {Path(model_params_path).as_posix()!r}",
        ]
    )
    return "\n".join(lines)


def build_contract_tail_cell_source() -> str:
    return r'''
# Injected by marvis v3 (tail). Validates the notebook contract.
import json as _rmc_json
from pathlib import Path as _RmcPath
import re as _rmc_re
import numpy as _rmc_np
import pandas as _rmc_pd

_RMC_SUPPORTED_ALGORITHM_TEXT = "xgb, lgb, lr, catboost, scorecard, dnn"
_RMC_ALLOWED_ALGORITHMS = {"xgb", "lgb", "lr", "catboost", "scorecard", "dnn"}
_RMC_ALGORITHM_ALIASES = {
    "xgb": "xgb",
    "xgbm": "xgb",
    "xgbmclassifier": "xgb",
    "xgboost": "xgb",
    "xgbclassifier": "xgb",
    "xgboostclassifier": "xgb",
    "xgboostxgbclassifier": "xgb",
    "xgboostsklearnxgbclassifier": "xgb",
    "lgb": "lgb",
    "lgbm": "lgb",
    "lightgbm": "lgb",
    "lighgbm": "lgb",
    "lgbclassifier": "lgb",
    "lgbmclassifier": "lgb",
    "lightgbmclassifier": "lgb",
    "lightgbmsklearnlgbmclassifier": "lgb",
    "lr": "lr",
    "logit": "lr",
    "logistic": "lr",
    "logisticregression": "lr",
    "logisticregressioncv": "lr",
    "sklearnlinearmodellogisticregression": "lr",
    "逻辑回归": "lr",
    "cat": "catboost",
    "catboost": "catboost",
    "catboostclassifier": "catboost",
    "catboostcatboostclassifier": "catboost",
    "scorecard": "scorecard",
    "scorecards": "scorecard",
    "评分卡": "scorecard",
    "记分卡": "scorecard",
    "dnn": "dnn",
    "deeplearning": "dnn",
    "deepneuralnetwork": "dnn",
    "neuralnetwork": "dnn",
    "mlp": "dnn",
    "mlpclassifier": "dnn",
    "神经网络": "dnn",
    "深度神经网络": "dnn",
}

def _rmc_algorithm_key(value):
    return _rmc_re.sub(r"[\W_]+", "", str(value).casefold(), flags=_rmc_re.UNICODE)

def _rmc_normalize_algorithm(value):
    key = str(value or "").strip()
    if not key:
        raise ValueError(
            "model algorithm is required; supported algorithms: "
            + _RMC_SUPPORTED_ALGORITHM_TEXT
        )
    if key in _RMC_ALLOWED_ALGORITHMS:
        return key
    normalized = _rmc_algorithm_key(key)
    if normalized in _RMC_ALGORITHM_ALIASES:
        return _RMC_ALGORITHM_ALIASES[normalized]
    raise ValueError(
        "unsupported model algorithm; supported algorithms: "
        + _RMC_SUPPORTED_ALGORITHM_TEXT
    )

if "RMC_SAMPLE_DF" not in globals():
    raise NameError("RMC_SAMPLE_DF must be defined as a pandas DataFrame at notebook top level")
if not isinstance(RMC_SAMPLE_DF, _rmc_pd.DataFrame):
    raise TypeError("RMC_SAMPLE_DF must be a pandas DataFrame")
if "RMC_SCORE_FN" not in globals() or not callable(RMC_SCORE_FN):
    raise NameError("RMC_SCORE_FN must be defined as a callable at notebook top level")
if "RMC_TARGET_COL" not in globals() or not isinstance(RMC_TARGET_COL, str):
    raise NameError("RMC_TARGET_COL must be defined as a string at notebook top level")

_rmc_params = globals().get("RMC_MODEL_PARAMS", globals().get("MODEL_HYPERPARAMETERS"))
if _rmc_params is not None:
    if not isinstance(_rmc_params, dict):
        raise TypeError("RMC_MODEL_PARAMS must be a dict")
    if not all(isinstance(key, str) for key in _rmc_params):
        raise TypeError("RMC_MODEL_PARAMS keys must be strings")

if "RMC_ALGORITHM" in globals():
    _rmc_algorithm_raw = RMC_ALGORITHM
elif isinstance(_rmc_params, dict) and "algorithm" in _rmc_params:
    _rmc_algorithm_raw = _rmc_params["algorithm"]
else:
    raise NameError("RMC_ALGORITHM must be defined as a supported model algorithm at notebook top level")
_rmc_algorithm = _rmc_normalize_algorithm(_rmc_algorithm_raw)

_rmc_sample = RMC_SAMPLE_DF.copy()
if RMC_TARGET_COL not in _rmc_sample.columns:
    raise ValueError(f"RMC_TARGET_COL={RMC_TARGET_COL!r} is not present in sample data")

_rmc_raw_scores = RMC_SCORE_FN(_rmc_sample.copy())
_rmc_score_series = _rmc_pd.Series(_rmc_raw_scores)
if len(_rmc_score_series) != len(_rmc_sample):
    raise ValueError(
        f"RMC_SCORE_FN returned {len(_rmc_score_series)} scores for {len(_rmc_sample)} rows"
    )
_rmc_score_series = _rmc_pd.to_numeric(_rmc_score_series, errors="raise")
if _rmc_score_series.isna().any():
    raise ValueError("RMC_SCORE_FN returned null scores")
if not _rmc_np.isfinite(_rmc_score_series.to_numpy(dtype=float)).all():
    raise ValueError("RMC_SCORE_FN returned non-finite scores")

_rmc_scores_path = _RmcPath(RMC_CODE_SCORES_PATH)
_rmc_scores_path.parent.mkdir(parents=True, exist_ok=True)
_rmc_pd.DataFrame({
    "row_index": range(len(_rmc_sample)),
    "code_model_score": _rmc_score_series.astype(float),
}).to_csv(_rmc_scores_path, index=False)

_rmc_runtime_sample_path = _RmcPath(RMC_RUNTIME_SAMPLE_PATH)
_rmc_runtime_sample_path.parent.mkdir(parents=True, exist_ok=True)
_rmc_sample.reset_index(drop=True).to_csv(_rmc_runtime_sample_path, index=False)

_rmc_importance_path = None
_rmc_importance = globals().get("RMC_FEATURE_IMPORTANCE", globals().get("FEATURE_IMPORTANCE"))
if _rmc_importance is not None:
    if not isinstance(_rmc_importance, _rmc_pd.DataFrame):
        raise TypeError("RMC_FEATURE_IMPORTANCE must be a pandas DataFrame")
    _rmc_missing_cols = {"feature", "importance"} - set(_rmc_importance.columns)
    if _rmc_missing_cols:
        raise ValueError("RMC_FEATURE_IMPORTANCE missing columns: " + ", ".join(sorted(_rmc_missing_cols)))
    _rmc_importance_cols = ["feature", "importance"]
    _rmc_category_col = None
    for _rmc_candidate_col in ("category", "类别"):
        if _rmc_candidate_col in _rmc_importance.columns:
            _rmc_category_col = _rmc_candidate_col
            _rmc_importance_cols.append(_rmc_candidate_col)
            break
    _rmc_importance_out = _rmc_importance[_rmc_importance_cols].copy()
    _rmc_importance_out["feature"] = _rmc_importance_out["feature"].astype(str)
    _rmc_importance_out["importance"] = _rmc_pd.to_numeric(
        _rmc_importance_out["importance"], errors="raise"
    )
    if _rmc_category_col is not None:
        _rmc_importance_out = _rmc_importance_out.rename(columns={_rmc_category_col: "category"})
        _rmc_importance_out["category"] = _rmc_importance_out["category"].fillna("").astype(str)
        _rmc_importance_out = _rmc_importance_out[["feature", "category", "importance"]]
    _rmc_importance_out = _rmc_importance_out.sort_values("importance", ascending=False)
    _rmc_importance_path = _RmcPath(RMC_FEATURE_IMPORTANCE_PATH)
    _rmc_importance_path.parent.mkdir(parents=True, exist_ok=True)
    _rmc_importance_out.to_csv(_rmc_importance_path, index=False)

_rmc_params_path = None
if _rmc_params is not None:
    _rmc_params_path = _RmcPath(RMC_MODEL_PARAMS_PATH)
    _rmc_params_path.parent.mkdir(parents=True, exist_ok=True)
    _rmc_params_path.write_text(
        _rmc_json.dumps(_rmc_params, ensure_ascii=False, default=str, indent=2),
        encoding="utf-8",
    )

_rmc_meta = {
    "contract_version": "rmc.v1",
    "target_col": RMC_TARGET_COL,
    "algorithm": _rmc_algorithm,
    "split_col": globals().get("RMC_SPLIT_COL"),
    "time_col": globals().get("RMC_TIME_COL"),
    "pmml_output_field": globals().get("RMC_PMML_OUTPUT_FIELD", "probability_1"),
    "score_decimal_places": int(globals().get("RMC_SCORE_DECIMAL_PLACES", 6)),
    "code_model_scores_path": str(_rmc_scores_path),
    "sample_snapshot_path": str(_rmc_runtime_sample_path),
    "feature_importance_path": str(_rmc_importance_path) if _rmc_importance_path else None,
    "model_params_path": str(_rmc_params_path) if _rmc_params_path else None,
}
_RmcPath(RMC_CONTRACT_META_PATH).parent.mkdir(parents=True, exist_ok=True)
_RmcPath(RMC_CONTRACT_META_PATH).write_text(
    _rmc_json.dumps(_rmc_meta, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
'''.strip()


def load_runtime_contract(path: Path) -> RuntimeContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RuntimeContract(
        target_col=str(payload["target_col"]),
        split_col=_optional_str(payload.get("split_col")),
        time_col=_optional_str(payload.get("time_col")),
        pmml_output_field=str(payload.get("pmml_output_field") or "probability_1"),
        score_decimal_places=int(
            6
            if payload.get("score_decimal_places") is None
            else payload["score_decimal_places"]
        ),
        code_model_scores_path=Path(payload["code_model_scores_path"]),
        sample_snapshot_path=_optional_path(payload.get("sample_snapshot_path")),
        feature_importance_path=_optional_path(payload.get("feature_importance_path")),
        model_params_path=_optional_path(payload.get("model_params_path")),
        algorithm=normalize_algorithm(payload.get("algorithm")),
    )


def _read_notebook_snapshot(notebook_or_path: Any):
    if isinstance(notebook_or_path, (str, Path)):
        path = Path(notebook_or_path).resolve()
        raw = path.read_bytes()
        notebook = read_notebook_bytes(raw, source=str(path), as_version=4)
        return notebook, (str(path), sha256(raw).hexdigest())
    return notebook_or_path, None


def _optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
