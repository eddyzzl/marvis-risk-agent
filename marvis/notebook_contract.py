from dataclasses import dataclass
import ast
import json
from pathlib import Path
from typing import Any

import nbformat

from marvis.model_algorithms import normalize_algorithm

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
    pass


@dataclass(frozen=True)
class ContractPrecheckResult:
    missing_names: list[str]
    target_col: str | None = None
    algorithm: str | None = None


@dataclass(frozen=True)
class _ContractBindings:
    sample_df_defined: bool = False
    score_fn_defined: bool = False
    target_col_defined: bool = False
    algorithm_defined: bool = False
    target_col: str | None = None
    algorithm: str | None = None
    invalid_names: tuple[str, ...] = ()


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
    algorithm: str = ""


def precheck_notebook_contract(notebook_or_path: Any) -> ContractPrecheckResult:
    notebook = _read_notebook(notebook_or_path)
    bindings = _contract_bindings(notebook)
    missing = []
    if not bindings.sample_df_defined:
        missing.append("RMC_SAMPLE_DF")
    if not bindings.score_fn_defined:
        missing.append("RMC_SCORE_FN")
    if not bindings.target_col_defined:
        missing.append("RMC_TARGET_COL")
    if not bindings.algorithm_defined:
        missing.append("RMC_ALGORITHM")
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
    return ContractPrecheckResult(
        missing_names=[],
        target_col=bindings.target_col,
        algorithm=bindings.algorithm,
    )


def _contract_bindings(notebook: Any) -> _ContractBindings:
    sample_df_defined = False
    score_fn_defined = False
    target_col_defined = False
    explicit_algorithm_defined = False
    fallback_algorithm_defined = False
    target_col: str | None = None
    algorithm: str | None = None
    invalid_names: list[str] = []
    fallback_invalid_names: list[str] = []

    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        tree = _parse_contract_scan_cell(str(cell.source), cell_index)
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
                    algorithm = _algorithm_literal(
                        "RMC_ALGORITHM",
                        node.value,
                        invalid_names,
                    )
                if not explicit_algorithm_defined and (
                    "RMC_MODEL_PARAMS" in target_names
                    or "MODEL_HYPERPARAMETERS" in target_names
                ):
                    invalid_count = len(fallback_invalid_names)
                    params_algorithm = _model_params_algorithm_literal(
                        node.value,
                        fallback_invalid_names,
                    )
                    if params_algorithm is not None or len(fallback_invalid_names) > invalid_count:
                        fallback_algorithm_defined = True
                        if params_algorithm is not None:
                            algorithm = params_algorithm
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
                    algorithm = _algorithm_literal(
                        "RMC_ALGORITHM",
                        node.value,
                        invalid_names,
                    )
                if (
                    not explicit_algorithm_defined
                    and node.value is not None
                    and (
                        "RMC_MODEL_PARAMS" in target_names
                        or "MODEL_HYPERPARAMETERS" in target_names
                    )
                ):
                    invalid_count = len(fallback_invalid_names)
                    params_algorithm = _model_params_algorithm_literal(
                        node.value,
                        fallback_invalid_names,
                    )
                    if params_algorithm is not None or len(fallback_invalid_names) > invalid_count:
                        fallback_algorithm_defined = True
                        if params_algorithm is not None:
                            algorithm = params_algorithm

    if not explicit_algorithm_defined and fallback_invalid_names:
        invalid_names.extend(fallback_invalid_names)

    return _ContractBindings(
        sample_df_defined=sample_df_defined,
        score_fn_defined=score_fn_defined,
        target_col_defined=target_col_defined,
        algorithm_defined=explicit_algorithm_defined or fallback_algorithm_defined,
        target_col=target_col,
        algorithm=algorithm,
        invalid_names=tuple(sorted(set(invalid_names))),
    )


def _parse_contract_scan_cell(source: str, cell_index: int) -> ast.Module | None:
    candidates = [source]
    notebook_python_source = _strip_ipython_syntax_for_static_scan(source)
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
            f"syntax error in RMC contract code cell {cell_index}: {last_error.msg}"
        ) from last_error
    return None


def _strip_ipython_syntax_for_static_scan(source: str) -> str:
    cleaned: list[str] = []
    for line_index, line in enumerate(source.splitlines()):
        stripped = line.lstrip()
        if line_index == 0 and stripped.startswith("%%"):
            continue
        if stripped.startswith(("%", "!")):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


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


def _algorithm_literal(
    name: str,
    value: ast.expr,
    invalid_names: list[str],
) -> str | None:
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        invalid_names.append(f"{name} must be a string literal")
        return None
    try:
        return normalize_algorithm(value.value)
    except ValueError as exc:
        invalid_names.append(f"{name} {exc}")
        return None


def _model_params_algorithm_literal(
    value: ast.expr,
    invalid_names: list[str],
) -> str | None:
    if not isinstance(value, ast.Dict):
        return None
    for key_node, value_node in zip(value.keys, value.values, strict=True):
        if (
            isinstance(key_node, ast.Constant)
            and key_node.value == "algorithm"
            and isinstance(value_node, ast.Constant)
        ):
            if not isinstance(value_node.value, str):
                invalid_names.append("RMC_MODEL_PARAMS algorithm must be a string")
                return None
            try:
                return normalize_algorithm(value_node.value)
            except ValueError as exc:
                invalid_names.append(f"RMC_MODEL_PARAMS algorithm {exc}")
                return None
    return None


def build_contract_head_cell_source(
    *,
    sample_path: Path,
    contract_meta_path: Path,
    code_scores_path: Path,
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
            f"RMC_SAMPLE_PATH = {Path(sample_path).as_posix()!r}",
            f"RMC_CONTRACT_META_PATH = {Path(contract_meta_path).as_posix()!r}",
            f"RMC_CODE_SCORES_PATH = {Path(code_scores_path).as_posix()!r}",
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
        score_decimal_places=int(payload.get("score_decimal_places") or 6),
        code_model_scores_path=Path(payload["code_model_scores_path"]),
        feature_importance_path=_optional_path(payload.get("feature_importance_path")),
        model_params_path=_optional_path(payload.get("model_params_path")),
        algorithm=normalize_algorithm(payload.get("algorithm")),
    )


def _read_notebook(notebook_or_path: Any):
    if isinstance(notebook_or_path, (str, Path)):
        return nbformat.read(notebook_or_path, as_version=4)
    return notebook_or_path


def _optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
