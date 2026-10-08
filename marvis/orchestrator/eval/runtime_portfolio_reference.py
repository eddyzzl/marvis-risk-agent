"""Independent portfolio arithmetic over authenticated source records.

No producer algorithms, output payloads, or producer-selected parameters are
accepted here. Callers must authenticate source bytes and bind the business
declaration separately. This calculator alone does not establish acceptance.
"""

from collections import defaultdict
import math
import re

from .runtime_archive_reader import _Unsupported


def segment_reference(records, *, segment_col, balance_col, top_k=20):
    """Full-panel count and EAD concentrations, before small groups are merged."""
    if not records or type(top_k) is not int or top_k < 1:
        raise ValueError("invalid segment reference declaration")
    groups = {}
    for row in records:
        segment, balance = row[segment_col], row[balance_col]
        if not isinstance(segment, str) or not segment:
            raise _Unsupported("ambiguous portfolio segment representation")
        if type(balance) not in {int, float} or not math.isfinite(balance) or balance < 0:
            raise ValueError("invalid segment exposure")
        group = groups.setdefault(segment, [0, 0.0])
        group[0] += 1
        group[1] += balance
    if sum(value[1] for value in groups.values()) <= 0:
        raise ValueError("empty portfolio exposure")

    def concentration(weights):
        weights = sorted(weights, reverse=True)
        shares = [weight / sum(weights) for weight in weights]
        return {"top1_pct": shares[0], "top5_pct": sum(shares[:5]),
                "hhi": sum(value * value for value in shares)}

    counts = concentration([value[0] for value in groups.values()])
    exposure = concentration([value[1] for value in groups.values()])
    ordered = sorted(groups, key=lambda key: -groups[key][0])
    kept, merged = ordered[:top_k], ordered[top_k:]
    if merged and "其他" in kept:
        raise _Unsupported("declared segment collides with aggregate display label")
    display = [(key, groups[key][0]) for key in kept]
    if merged:
        display.append(("其他", sum(groups[key][0] for key in merged)))
    flags = []
    if counts["top1_pct"] > 0.4 or counts["hhi"] > 0.25:
        flags.append({"kind": "high_concentration", "top1_pct": counts["top1_pct"], "hhi": counts["hhi"]})
    if merged:
        flags.append({"kind": "sparse_segment", "merged_count": len(merged)})
    return {
        "segments": [{"segment": key, "count": count, "pop_pct": count / len(records),
                      "approval_rate": None, "bad_rate": None, "avg_score": None, "net_profit": None}
                     for key, count in display],
        "concentration": counts, "concentration_basis": "count",
        "ead_concentration": exposure, "ead_concentration_basis": "ead", "red_flags": flags,
    }


def reference_flag_text(flag, *, top_k=20):
    """Standard report wording for independently established flag values."""
    kind = flag["kind"]
    if kind == "sparse_month":
        return f"月 {flag['month']} 对齐月对仅 {flag['pair_count']} 对（<100），转移率不稳健。"
    if kind == "month_gap":
        return (f"快照 {flag['month']} 至 {flag['to_month']} 间隔 {flag['interval_months']} 个月；"
                "该转移是跨期观测，不能作为单月迁移概率用于月度预期损失。")
    if kind == "matrix_not_absorbing":
        return (f"损失态 `{flag['loss_state']}` 在观测迁徙矩阵中并非吸收态（有迁出概率）；"
                "已按吸收态强制处理（该行自环=1）后估计，结果偏保守下限。")
    if kind == "short_history":
        return f"可用快照月仅 {flag['months_available']} 个（<3），迁徙矩阵估计不稳健。"
    if kind == "high_concentration":
        return (f"细分高度集中：top1 占比 {flag['top1_pct']:.1%}，HHI {flag['hhi']:.3f}"
                "（阈值 top1>40% 或 HHI>0.25）。")
    if kind == "sparse_segment":
        return f"{flag['merged_count']} 个小细分已归并为「其他」（top_k={top_k}）。"
    raise _Unsupported("unrecognized independent portfolio flag")


def transition_reference(records, *, id_col, snapshot_col, bucket_col,
                         balance_col, states, loss_state, lgd, horizon_months,
                         window=None):
    """Count transitions by snapshot, then propagate loss probabilities as a vector.

    Canonical YYYY-MM snapshots only. Other source representations need a
    separately reviewed decoder; ambiguity is never converted into a pass.
    Returned flags contain semantic values, excluding presentation prose.
    """
    states = list(states)
    if (not states or len(set(states)) != len(states)
            or any(not isinstance(s, str) or not s for s in states)
            or {"exited", "from"}.intersection(states) or loss_state not in states):
        raise ValueError("invalid portfolio state declaration")
    if (type(lgd) not in {int, float} or not math.isfinite(lgd) or not 0 <= lgd <= 1
            or type(horizon_months) is not int or not 1 <= horizon_months <= 1200):
        raise ValueError("invalid portfolio loss declaration")
    if len(records) > 1_000_000 or len(states) > 100:
        raise _Unsupported("portfolio independent computation budget exceeded")
    panel = defaultdict(dict)
    id_types = set()
    for row in records:
        loan, month, state = row[id_col], row[snapshot_col], row[bucket_col]
        if type(loan) not in {str, int} or loan == "":
            raise _Unsupported("ambiguous portfolio loan identity")
        id_types.add(type(loan))
        if len(id_types) != 1:
            raise _Unsupported("mixed portfolio loan identity representations")
        if not isinstance(month, str) or not re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", month) or month[:4] == "0000":
            raise _Unsupported("portfolio reference requires canonical calendar months")
        weight = row[balance_col]
        if type(weight) not in {int, float} or not math.isfinite(weight) or weight < 0:
            raise ValueError("invalid portfolio balance")
        if state not in states or loan in panel[month]:
            raise ValueError("unknown state or duplicate loan snapshot")
        panel[month][loan] = (state, float(weight))
    months = sorted(panel)
    if len(months) < 2:
        raise _Unsupported("no observed portfolio transition")
    selected = months[:-1] if not window else [m for m in months[:-1] if m in window]
    if window and set(window) - set(months[:-1]):
        raise ValueError("unavailable portfolio migration window")
    destinations = states + ["exited"]
    weighted_matrices, count_matrices, net_flows, flags = [], {}, [], []
    for month, next_month in zip(months, months[1:]):
        year, number = map(int, month.split("-"))
        next_year, next_number = map(int, next_month.split("-"))
        interval = 12 * (next_year - year) + next_number - number
        if interval != 1:
            flags.append({"kind": "month_gap", "month": month, "to_month": next_month, "interval_months": interval})
            if month in selected:
                raise ValueError("non-monthly portfolio loss migration")
        amounts, counts = defaultdict(float), defaultdict(int)
        into_bad = out_of_bad = 0.0
        for loan, (origin, weight) in panel[month].items():
            target = panel[next_month].get(loan, ("exited", 0))[0]
            amounts[origin, target] += weight
            counts[origin, target] += 1
            if origin != states[-1] and target == states[-1]:
                into_bad += weight
            if origin == states[-1] and target not in {states[-1], "exited"}:
                out_of_bad += weight

        def normalized(cells):
            result = []
            for origin in states:
                row = [cells[origin, target] for target in destinations]
                total = sum(row)
                result.append([value / total if total else 0.0 for value in row])
            return result

        count_matrices[month] = normalized(counts)
        weighted_matrices.append({
            "month": month, "to_month": next_month,
            "from_to_matrix": normalized(amounts),
            "base": {state: sum(amounts[state, dest] for dest in destinations) for state in states},
            "base_kind": "balance", "pair_count": len(panel[month]),
        })
        net_flows.append({"month": month, "into_bad": into_bad, "out_of_bad": out_of_bad})
        if len(panel[month]) < 100:
            flags.append({"kind": "sparse_month", "month": month, "pair_count": len(panel[month])})
    observed = [item["from_to_matrix"] for item in weighted_matrices if item["month"] in selected]
    average = [[sum(matrix[i][j] for matrix in observed) / len(observed)
                for j in range(len(destinations))] for i in range(len(states))]
    worst = [[max(matrix[i][j] for matrix in observed)
              for j in range(len(destinations))] for i in range(len(states))]
    transition = []
    for i in range(len(states)):
        row = [sum(count_matrices[month][i][j] for month in selected) / len(selected)
               for j in range(len(states))]
        total = sum(row)
        transition.append([value / total for value in row] if total else
                          [float(i == j) for j in range(len(states))])
    loss_index = states.index(loss_state)
    was_absorbing = transition[loss_index][loss_index] >= 1 - 1e-9
    transition[loss_index] = [float(j == loss_index) for j in range(len(states))]
    probability = [float(j == loss_index) for j in range(len(states))]
    # Backward recurrence uses a vector; producer uses full matrix powers.
    for _ in range(horizon_months):
        probability = [sum(weight * p for weight, p in zip(row, probability)) for row in transition]
    by_state = dict(zip(states, probability))
    by_month = [{
        "month": month, "balance": sum(weight for _, weight in panel[month].values()),
        "expected_loss": sum(weight * by_state[state] * lgd for state, weight in panel[month].values()),
        "is_reference": month == months[-1],
    } for month in months]
    loss_flags = []
    if not was_absorbing:
        loss_flags.append({"kind": "matrix_not_absorbing", "loss_state": loss_state})
    if len(months) < 3:
        loss_flags.append({"kind": "short_history", "months_available": len(months)})
    return {
        "flow": {"states": states, "to_states": destinations, "months": months[:-1],
                 "matrix_by_month": weighted_matrices, "net_flows": net_flows, "red_flags": flags},
        "migration": {"states": states, "to_states": destinations, "window_months": selected,
                      "avg_matrix": average, "worst_matrix": worst,
                      "heat_table": [{"from": state, **dict(zip(destinations, row))}
                                     for state, row in zip(states, average)], "red_flags": flags},
        "expected_loss": {"loss_state": loss_state, "chain": [
            {"from_state": state, "p_to_loss": by_state[state]} for state in states],
            "el_by_month": by_month, "total_el": by_month[-1]["expected_loss"],
            "assumptions": {"lgd": float(lgd), "horizon_months": horizon_months,
                            "matrix_window": selected, "loss_state": loss_state,
                            "total_el_basis": "reference_snapshot", "reference_snapshot": months[-1]},
            "red_flags": loss_flags},
    }
