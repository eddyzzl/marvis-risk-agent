"""确定性 0/1 坏标签构造内核 (C1 标签构造与成熟度工具).

信贷风控建模的最重要前置：从一张"贷款×期"的还款/DPD 长表，按业务给定的
**观察期 / 表现期 / 定坏口径** 构造一个 0/1 目标（坏=1）。本模块是纯函数内核
（不触 IO、不触子进程），给定同一输入必产出同一输出（INV-1 确定性）。

输入长表（DPD 长表）契约——沿用 ``marvis.data.performance`` 已有的"表现期快照"列
约定，一行 = 一笔贷款在某个 MOB（month-on-book）上的逾期状态：

- ``id_col``：贷款唯一键。
- ``mob_col``：账龄 MOB（非负整数；0=放款当期）。
- 逾期强度二选一：
  - ``dpd_col``：数值逾期天数（days past due）；口径阈值 ``threshold_dpd`` 是天数。
  - ``status_col`` + ``states``：离散逾期桶（如 C/M1/M2/M3+），桶的"由好到坏"
    语义顺序由调用方通过 ``states`` 显式给定（机器不猜，与 performance/roll_rate 一致）；
    口径阈值 ``threshold_status`` 是 ``states`` 中的某个桶，命中=达到或坏于该桶。
- ``cohort_col``（可选，成熟度检查用）：放款 vintage（YYYY-MM），判定表现期是否闭合。

定坏口径（bad definition）三要素：

- 观察期 ``observation_window``：观察窗口末端 MOB。特征在此点可知；标签只看此点之后。
- 表现期 ``performance_window``：观察点之后再往后看多少个 MOB 判定是否变坏。
- 逾期阈值：``threshold_dpd``（如 30/60/90 天）或 ``threshold_status``（如 "M2"）；
  可选 ``at_mob`` 指定在哪个 MOB 判定（如 "90+@mob6"），缺省=表现期末端。

一笔贷款的标签：在表现期窗口 ``(observation_window, at_mob]`` 内**曾经**达到定坏阈值
→ 1；全程未达到 → 0；表现期未观测完整（缺该窗口内足够的 MOB 观测）→ 不可定坏，
标签为 NaN（不静默当好客户；下游 NaN 标签门决定丢弃或补数据）。

输出携带**定坏口径元数据** ``BadDefinition``，进 T3 血缘（NumberProvenance 的 params）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from marvis.validation.vintage import _cohort_key


@dataclass(frozen=True)
class BadDefinition:
    """定坏口径元数据（进 T3 血缘的 params 一部分；确定性可复算）。"""

    #: 逾期强度口径："dpd"（数值天数）或 "status"（离散桶）。
    threshold_kind: str
    #: 阈值：dpd 口径为天数字符串化的数值；status 口径为桶名（states 内）。
    threshold: float | str
    #: 观察窗口末端 MOB（特征可知点；标签只看此后）。
    observation_window: int
    #: 表现期长度（观察点后再看的 MOB 数）。
    performance_window: int
    #: 判定 MOB（如 90+@mob6 的 6）。缺省 = observation_window + performance_window。
    at_mob: int
    #: 命中定义："ever"（表现期窗口内曾达标即坏）；当前唯一支持的口径。
    hit_rule: str = "ever"

    def to_dict(self) -> dict:
        return {
            "threshold_kind": self.threshold_kind,
            "threshold": self.threshold,
            "observation_window": self.observation_window,
            "performance_window": self.performance_window,
            "at_mob": self.at_mob,
            "hit_rule": self.hit_rule,
            "label": self.label_expression(),
        }

    def label_expression(self) -> str:
        """人读的定坏口径表达式，如 ``90+@mob6 (obs=0, perf=6)``。"""
        if self.threshold_kind == "dpd":
            head = f"DPD{int(self.threshold)}+@mob{self.at_mob}"
        else:
            head = f"{self.threshold}+@mob{self.at_mob}"
        return f"{head} (obs={self.observation_window}, perf={self.performance_window})"


@dataclass(frozen=True)
class LabelConstruction:
    """标签构造结果：每笔贷款一行的 0/1（或 NaN）目标 + 口径元数据 + 计数。"""

    #: 一行一贷款：id_col + cohort（若给定）+ target 列（0/1/NaN）。
    frame: pd.DataFrame
    #: target 列名。
    target_col: str
    definition: BadDefinition
    n_loans: int
    n_bad: int
    n_good: int
    #: 表现期未闭合、无法定坏的贷款数（target=NaN）。
    n_unmatured: int


def _resolve_threshold_kind(
    *,
    dpd_col: str | None,
    status_col: str | None,
    threshold_dpd,
    threshold_status,
    states,
) -> tuple[str, str | None, tuple[str, ...]]:
    """确定逾期强度口径，返回 ``(kind, resolved_col, state_order)``。"""
    has_dpd = dpd_col is not None and threshold_dpd is not None
    has_status = status_col is not None and threshold_status is not None
    if has_dpd and has_status:
        raise ValueError(
            "同时给定了 dpd 口径 (dpd_col+threshold_dpd) 与 status 口径 "
            "(status_col+threshold_status)；只能二选一。"
        )
    if not has_dpd and not has_status:
        raise ValueError(
            "缺少逾期强度口径：请给 dpd_col+threshold_dpd（数值天数）"
            "或 status_col+threshold_status+states（离散桶）。"
        )
    if has_dpd:
        return "dpd", str(dpd_col), ()
    state_order = tuple(str(state) for state in (states or ()))
    if not state_order:
        raise ValueError("status 口径必须提供 states（桶的由好到坏顺序）。")
    if len(set(state_order)) != len(state_order):
        raise ValueError("states 含重复桶取值；每个桶只能出现一次。")
    if str(threshold_status) not in state_order:
        raise ValueError(
            f"threshold_status={threshold_status!r} 不在 states={list(state_order)} 内。"
        )
    return "status", str(status_col), state_order


def _hit_mask_dpd(values: pd.Series, threshold: float) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    # NaN DPD（该期无观测）不算命中；达到或超过阈值天数即命中。
    return np.where(np.isfinite(numeric), numeric >= float(threshold), False)


def _hit_mask_status(
    values: pd.Series, threshold: str, state_order: tuple[str, ...]
) -> np.ndarray:
    # 命中 = 桶坏于或等于阈值桶（在 states 由好到坏顺序里索引 >= 阈值索引）。
    rank = {state: index for index, state in enumerate(state_order)}
    threshold_rank = rank[str(threshold)]
    observed = values.map(lambda v: rank.get(str(v)) if pd.notna(v) else None)
    return np.array(
        [False if r is None else (r >= threshold_rank) for r in observed.tolist()],
        dtype=bool,
    )


def construct_label(
    df: pd.DataFrame,
    *,
    id_col: str,
    mob_col: str,
    observation_window: int,
    performance_window: int,
    dpd_col: str | None = None,
    threshold_dpd: float | None = None,
    status_col: str | None = None,
    threshold_status: str | None = None,
    states: list[str] | tuple[str, ...] | None = None,
    at_mob: int | None = None,
    cohort_col: str | None = None,
    target_col: str = "target",
) -> LabelConstruction:
    """从 DPD 长表构造 0/1 坏标签（见模块 docstring 的完整口径说明）。

    确定性：给定同一 ``df`` 与同一组口径参数，产出逐位一致的结果。
    """
    if observation_window < 0:
        raise ValueError("observation_window must be non-negative")
    if performance_window < 1:
        raise ValueError("performance_window must be >= 1")
    resolved_at_mob = observation_window + performance_window if at_mob is None else int(at_mob)
    if resolved_at_mob <= observation_window:
        raise ValueError("at_mob must be greater than observation_window")

    kind, value_col, state_order = _resolve_threshold_kind(
        dpd_col=dpd_col,
        status_col=status_col,
        threshold_dpd=threshold_dpd,
        threshold_status=threshold_status,
        states=states,
    )
    required = [id_col, mob_col, value_col]
    if cohort_col:
        required.append(cohort_col)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")

    work = df[required].copy()
    work["_marvis_mob"] = pd.to_numeric(work[mob_col], errors="coerce")
    finite_mob = work["_marvis_mob"].dropna()
    if ((finite_mob < 0) | (finite_mob % 1 != 0)).any():
        raise ValueError("MOB must be non-negative integers")
    work = work[work["_marvis_mob"].notna()].copy()
    work["_marvis_mob"] = work["_marvis_mob"].astype(int)

    if kind == "dpd":
        work["_marvis_hit"] = _hit_mask_dpd(work[value_col], float(threshold_dpd))
    else:
        work["_marvis_hit"] = _hit_mask_status(work[value_col], str(threshold_status), state_order)

    # 表现期窗口 = 观察点之后到判定 MOB：(observation_window, at_mob]。
    in_window = (work["_marvis_mob"] > observation_window) & (work["_marvis_mob"] <= resolved_at_mob)
    window = work[in_window]

    # 每笔贷款：表现期窗口内曾命中 -> 坏；观测到判定 MOB 或更远且从未命中 -> 好；
    # 未观测到判定 MOB（表现期未闭合）-> 不可定坏 (NaN)。
    # 命中判定只看表现期窗口 (observation_window, at_mob] 内的行；
    # 成熟度判定用该 loan **全量**（未被 at_mob 上界截断）的最大观测 MOB —— 观测到
    # at_mob 或之后即成熟，不因 at_mob 那一期恰好缺行（月度服务文件常见缺失）而误判未成熟。
    ids = pd.Index(pd.unique(work[id_col]))
    ever_hit = window.groupby(id_col)["_marvis_hit"].any()
    max_mob_observed = work.groupby(id_col)["_marvis_mob"].max()

    target_values: list[float] = []
    n_bad = n_good = n_unmatured = 0
    for loan_id in ids:
        hit = bool(ever_hit.get(loan_id, False))
        matured = int(max_mob_observed.get(loan_id, -1)) >= resolved_at_mob
        if hit:
            target_values.append(1.0)
            n_bad += 1
        elif matured:
            target_values.append(0.0)
            n_good += 1
        else:
            target_values.append(float("nan"))
            n_unmatured += 1

    out = pd.DataFrame({id_col: list(ids)})
    if cohort_col:
        cohort_by_id = work.drop_duplicates(subset=[id_col]).set_index(id_col)[cohort_col]
        out[cohort_col] = out[id_col].map(cohort_by_id)
    out[target_col] = target_values

    definition = BadDefinition(
        threshold_kind=kind,
        threshold=float(threshold_dpd) if kind == "dpd" else str(threshold_status),
        observation_window=int(observation_window),
        performance_window=int(performance_window),
        at_mob=int(resolved_at_mob),
    )
    return LabelConstruction(
        frame=out,
        target_col=target_col,
        definition=definition,
        n_loans=int(len(ids)),
        n_bad=int(n_bad),
        n_good=int(n_good),
        n_unmatured=int(n_unmatured),
    )


@dataclass(frozen=True)
class CohortMaturity:
    """单个放款 cohort 的表现期成熟度判定。"""

    cohort: str
    n_loans: int
    #: 该 cohort 内观测到的最大 MOB（跨该 cohort 所有贷款）。
    max_observed_mob: int
    #: 定坏所需的 MOB（= at_mob）。
    required_mob: int
    matured: bool


@dataclass(frozen=True)
class MaturityReport:
    """按 cohort 的成熟度报告：哪些 cohort 表现期未闭合。"""

    required_mob: int
    cohorts: tuple[CohortMaturity, ...]
    immature_cohorts: tuple[str, ...]

    @property
    def all_matured(self) -> bool:
        return len(self.immature_cohorts) == 0


def check_cohort_maturity(
    df: pd.DataFrame,
    *,
    id_col: str,
    mob_col: str,
    cohort_col: str,
    required_mob: int,
) -> MaturityReport:
    """按 vintage cohort 判定表现期是否闭合到足以定坏。

    一个 cohort 成熟 = 其贷款观测到的最大 MOB >= ``required_mob``（定坏判定点）。
    未成熟 cohort 的坏标签会低估坏率（还没坏够就被当好），必须走确认门，不静默纳入。
    确定性：给定同输入必产出同 report。
    """
    required = [id_col, mob_col, cohort_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    work = df[required].copy()
    work["_marvis_mob"] = pd.to_numeric(work[mob_col], errors="coerce")
    work = work[work["_marvis_mob"].notna()].copy()
    work["_marvis_mob"] = work["_marvis_mob"].astype(int)
    work["_marvis_cohort"] = work[cohort_col].map(_cohort_key)

    cohorts: list[CohortMaturity] = []
    immature: list[str] = []
    for cohort, group in work.groupby("_marvis_cohort", sort=True):
        max_mob = int(group["_marvis_mob"].max())
        n_loans = int(group[id_col].nunique())
        matured = max_mob >= int(required_mob)
        cohorts.append(
            CohortMaturity(
                cohort=str(cohort),
                n_loans=n_loans,
                max_observed_mob=max_mob,
                required_mob=int(required_mob),
                matured=matured,
            )
        )
        if not matured:
            immature.append(str(cohort))
    return MaturityReport(
        required_mob=int(required_mob),
        cohorts=tuple(cohorts),
        immature_cohorts=tuple(immature),
    )


@dataclass(frozen=True)
class BadDefinitionSuggestion:
    """从 roll_rate 矩阵推出的定坏口径建议（tool_define_label 的推荐默认值）。"""

    threshold_status: str
    at_mob: int
    #: 该桶在建议 MOB 后的回滚率（roll-back 到更好状态的占比）。
    roll_back_rate: float
    rationale: str

    def to_dict(self) -> dict:
        return {
            "threshold_status": self.threshold_status,
            "at_mob": self.at_mob,
            "roll_back_rate": self.roll_back_rate,
            "rationale": self.rationale,
        }


#: 回滚率低于该阈值 = 该逾期桶基本"回不来了"，可作定坏点。
_STABLE_ROLL_BACK_THRESHOLD = 0.10


def suggest_bad_definition(
    *,
    states: list[str] | tuple[str, ...],
    matrix: list[list[float]] | tuple[tuple[float, ...], ...],
    at_mob: int,
    roll_back_threshold: float = _STABLE_ROLL_BACK_THRESHOLD,
) -> BadDefinitionSuggestion | None:
    """从既有 roll_rate_matrix 输出生成定坏口径建议。

    桥接逻辑：``states`` 是由好到坏排序的桶，``matrix[i][j]`` 是从 states[i] 转移到
    states[j] 的占比。对每个逾期桶（非首个"好"桶）算其"回滚率"= 转移到任何**更好**
    桶的占比之和。回滚率低（< 阈值）意味着进入该桶后基本不会回好——是稳定的定坏点。
    选回滚率低于阈值里**最靠前（最轻）**的逾期桶作建议（越早定坏越能扩样本），
    如"60+ 在 mob6 后回滚率 <10% → 建议 60+@mob6"。无满足者返回 None。

    确定性：纯函数，给定同 matrix 必产出同建议。
    """
    state_order = tuple(str(state) for state in states)
    n = len(state_order)
    if n < 2:
        return None
    rows = [tuple(float(value) for value in row) for row in matrix]
    if len(rows) != n or any(len(row) != n for row in rows):
        raise ValueError("matrix shape must match states length")

    # 逾期桶 = 索引 >= 1（索引 0 是最好的"当前/正常"桶）。
    for index in range(1, n):
        roll_back = sum(rows[index][j] for j in range(0, index))
        if roll_back < float(roll_back_threshold):
            return BadDefinitionSuggestion(
                threshold_status=state_order[index],
                at_mob=int(at_mob),
                roll_back_rate=float(roll_back),
                rationale=(
                    f"{state_order[index]} 在 mob{at_mob} 后回滚率 "
                    f"{roll_back * 100:.1f}% < {roll_back_threshold * 100:.0f}%，"
                    f"进入该桶后基本不再回好，建议以 {state_order[index]}@mob{at_mob} 定坏。"
                ),
            )
    return None


__all__ = [
    "BadDefinition",
    "BadDefinitionSuggestion",
    "CohortMaturity",
    "LabelConstruction",
    "MaturityReport",
    "check_cohort_maturity",
    "construct_label",
    "suggest_bad_definition",
]
