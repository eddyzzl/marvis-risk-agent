"""Independent row-wise typed DSL and population/economics/stability arithmetic.

No platform evaluator, backtest, candidate or economic calculator is imported.
This computes the declared software policy, not its institutional suitability.
"""
from collections import Counter
import json
import math
from numbers import Real
import statistics

import pandas as pd

from .runtime_archive_reader import _Unsupported
from .runtime_modeling_reference import psi_reference


def _missing(v):
    return v is None or bool(pd.isna(v))


def _numeric(v):
    return isinstance(v, Real) and not isinstance(v, bool)


def _strict(a, b):
    if isinstance(b, bool):
        return isinstance(a, bool) and a is b
    if _numeric(b):
        return _numeric(a) and a == b
    return isinstance(a, str) and isinstance(b, str) and a == b if isinstance(b, str) else a is b


def expression_reference(row, expression, *, depth=0):
    if depth > 32:
        raise _Unsupported("DSL expression depth limit")
    op = expression.get("op")
    if op in {"and", "or", "n_of_k"}:
        args = expression["args"]
        if not args or len(args) > 256:
            raise _Unsupported("DSL branch limit")
        if op == "and":
            return all(expression_reference(row, e, depth=depth + 1) for e in args)
        if op == "or":
            return any(expression_reference(row, e, depth=depth + 1) for e in args)
        n = expression["n"]
        if type(n) is not int or not 1 <= n <= len(args):
            raise ValueError("invalid cardinality")
        matched = 0
        for e in args:
            matched += expression_reference(row, e, depth=depth + 1)
            if matched >= n:
                return True
        return False
    if op == "not":
        return not expression_reference(row, expression["arg"], depth=depth + 1)
    field = expression["field"]
    if field not in row:
        raise ValueError("DSL field not found")
    value = row[field]
    missing = _missing(value)
    if op in {"is_null", "is_not_null"}:
        return missing if op == "is_null" else not missing
    if missing:
        policy = expression.get("missing", "no_match")
        if policy == "error":
            raise ValueError("DSL missing input refused")
        if policy not in {"match", "no_match"}:
            raise ValueError("unknown DSL missing policy")
        return policy == "match"
    if op == "between":
        low, high = expression["lower"], expression["upper"]
        if _numeric(low) and _numeric(high):
            value = float(value)
        return ((value >= low if expression.get("include_lower", True) else value > low)
                and (value <= high if expression.get("include_upper", True) else value < high))
    if op != "compare":
        raise _Unsupported("unknown DSL expression requires an independent evaluator")
    operator, expected = expression["operator"], expression["value"]
    if operator in {"in", "not_in"}:
        if not isinstance(expected, list):
            raise ValueError("membership literals must be list")
        if expression.get("coercion") == "strict":
            matched = any(_strict(value, v) for v in expected)
        else:
            if expected and all(_numeric(v) for v in expected):
                value = float(value)
            matched = value in expected
        return matched if operator == "in" else not matched
    if expression.get("coercion") == "strict":
        if operator not in {"==", "!="}:
            raise ValueError("strict coercion supports equality only")
        matched = _strict(value, expected)
        return matched if operator == "==" else not matched
    if _numeric(expected):
        value = float(value)
    comparisons = {"==": lambda: value == expected, "!=": lambda: value != expected,
        "<": lambda: value < expected, "<=": lambda: value <= expected,
        ">": lambda: value > expected, ">=": lambda: value >= expected}
    if operator not in comparisons:
        raise ValueError("unknown comparison operator")
    return bool(comparisons[operator]())


def decisions_reference(rows, spec):
    if spec.get("schema_version") not in {"strategy.dsl.v1", "strategy.dsl.v2"} or spec.get("match_policy") != "first_match":
        raise _Unsupported("strategy DSL schema/policy requires separate evaluator")
    rules = sorted(spec["rules"], key=lambda r: (r["priority"], r["rule_id"]))
    if len(rules) > 4096 or len(rows) > 100_000:
        raise _Unsupported("strategy independent row/rule limit")
    result = []
    for row in rows:
        chosen, rule_id = spec["default_action"], None
        for rule in rules:
            if expression_reference(row, rule["condition"]):
                chosen, rule_id = rule["action"], rule["rule_id"]
                break
        if chosen.get("stop", True) is not True:
            raise ValueError("non-stopping action unsupported")
        kind = chosen["type"]
        value = chosen.get("value", {"approval": "approve", "reject": "reject", "review": "review"}.get(kind))
        if kind in {"limit", "pricing"} and (not _numeric(value) or not math.isfinite(value) or value < 0 or kind == "pricing" and value > 1):
            raise ValueError("invalid typed numeric action")
        if kind not in {"approval", "reject", "review", "limit", "pricing", "segment", "collection"}:
            raise ValueError("unknown typed action")
        result.append({"matched_rule_id": rule_id, "action_type": kind, "value": value,
                       "decision": chosen.get("output_value", value), "reason_code": chosen.get("reason_code")})
    return result


def typed_impact_reference(rows, spec, target, *, baseline=None, economics=None, bad_value=1):
    decisions = decisions_reference(rows, spec)
    labels = [None if _missing(row[target]) else int(row[target] == bad_value) for row in rows]
    if any(not _missing(row[target]) and row[target] not in {0, 1} for row in rows):
        raise ValueError("typed impact binary target required")
    n, labeled = len(rows), sum(v is not None for v in labels)
    def ratio(a, b):
        return a / b if b else None
    metrics, breakdown, economic, transitions = {}, [], {}, []
    kind = spec["strategy_type"]
    overall = ratio(sum(v == 1 for v in labels), labeled)
    values = [d["value"] for d in decisions]
    if kind in {"approval", "reject"}:
        actions = [{"approval": "approve", "reject": "reject", "review": "review"}[d["action_type"]] for d in decisions]
        metrics.update(overall_bad_count=sum(v == 1 for v in labels), overall_bad_rate=overall)
        for action in ("approve", "reject", "review"):
            positions = [i for i, v in enumerate(actions) if v == action]
            count, count_labeled = len(positions), sum(labels[i] is not None for i in positions)
            bad = sum(labels[i] == 1 for i in positions)
            row = {"action": action, "count": count, "rate": count / n if n else 0.0,
                   "labeled_count": count_labeled, "bad_count": bad, "bad_rate": ratio(bad, count_labeled)}
            breakdown.append(row)
            metrics.update({f"{action}_{key}": value for key, value in row.items() if key != "action"})
        if kind == "reject":
            metrics.update(bad_capture_rate=ratio(sum(a == "reject" and y == 1 for a, y in zip(actions, labels, strict=True)), sum(y == 1 for y in labels)),
                good_reject_rate=ratio(sum(a == "reject" and y == 0 for a, y in zip(actions, labels, strict=True)), sum(y == 0 for y in labels)))
        if baseline is not None:
            previous = [{"approval": "approve", "reject": "reject", "review": "review"}[d["action_type"]]
                        for d in decisions_reference(rows, baseline)]
            for old in ("approve", "reject", "review"):
                for new in ("approve", "reject", "review"):
                    positions = [i for i, pair in enumerate(zip(previous, actions, strict=True)) if pair == (old, new)]
                    count_labeled = sum(labels[i] is not None for i in positions)
                    bad = sum(labels[i] == 1 for i in positions)
                    transitions.append({"from_action": old, "to_action": new, "count": len(positions),
                        "rate": ratio(len(positions), previous.count(old)), "population_share": len(positions) / n if n else 0.0,
                        "labeled_count": count_labeled, "bad_count": bad, "bad_rate": ratio(bad, count_labeled)})
    elif kind in {"limit", "pricing", "segmentation"}:
        tokens = {json.dumps(v, ensure_ascii=False, allow_nan=False): v for v in values}
        ordered = sorted(tokens, key=lambda token: tokens[token]) if kind in {"limit", "pricing"} else sorted(tokens)
        for token in ordered:
            value = tokens[token]
            positions = [i for i, v in enumerate(values) if json.dumps(v, ensure_ascii=False, allow_nan=False) == token]
            count_labeled = sum(labels[i] is not None for i in positions)
            bad = sum(labels[i] == 1 for i in positions)
            row = {"segment" if kind == "segmentation" else "assigned_limit" if kind == "limit" else "assigned_rate": value,
                "count": len(positions), "share": len(positions) / n, "labeled_count": count_labeled,
                "bad_count": bad, "bad_rate": ratio(bad, count_labeled)}
            if kind == "segmentation":
                row["lift"] = None if row["bad_rate"] is None or not overall else row["bad_rate"] / overall
            breakdown.append(row)
        if kind == "segmentation":
            metrics.update(segment_count=len(breakdown), overall_bad_count=sum(v == 1 for v in labels), overall_bad_rate=overall)
            if baseline is not None:
                previous = [d["value"] for d in decisions_reference(rows, baseline)]
                old_tokens = {json.dumps(v, ensure_ascii=False, allow_nan=False): v for v in previous}
                for old_token in sorted(old_tokens):
                    for new_token in sorted(tokens):
                        old, new = old_tokens[old_token], tokens[new_token]
                        count = sum(json.dumps(a, ensure_ascii=False, allow_nan=False) == old_token
                                    and json.dumps(b, ensure_ascii=False, allow_nan=False) == new_token
                                    for a, b in zip(previous, values, strict=True))
                        old_count = sum(json.dumps(v, ensure_ascii=False, allow_nan=False) == old_token for v in previous)
                        transitions.append({"from_segment": old, "to_segment": new, "count": count,
                            "rate": ratio(count, old_count), "population_share": count / n if n else 0.0})
        elif kind == "limit":
            metrics.update(count=n, total_limit=math.fsum(values), mean_limit=statistics.mean(values) if n else None,
                           min_limit=min(values) if n else None, max_limit=max(values) if n else None)
        else:
            metrics.update(count=n, mean_rate=statistics.mean(values) if n else None)
        if kind in {"limit", "pricing"}:
            old = None if baseline is None else [d["value"] for d in decisions_reference(rows, baseline)]
            prefix = "" if kind == "limit" else "repriced_"
            for direction, predicate in (("up", lambda a, b: a > b), ("down", lambda a, b: a < b), ("unchanged", lambda a, b: a == b)):
                key = f"{prefix if direction != 'unchanged' else ''}{direction}_count"
                metrics[key] = None if old is None else sum(predicate(a, b) for a, b in zip(values, old, strict=True))
            if kind == "limit":
                metrics["total_limit_delta"] = None if old is None else math.fsum(values) - math.fsum(old)
            if old is not None:
                directions = ("up", "down", "unchanged") if kind == "limit" else ("repriced_up", "repriced_down", "unchanged")
                transitions = [{"direction": direction, "count": metrics[f"{direction}_count"],
                                "rate": metrics[f"{direction}_count"] / n if n else 0.0} for direction in directions]
            if economics:
                economic = economics_reference(rows, values, kind, economics, old=old)
    else:
        raise _Unsupported("collection action economics require its declared domain adapter")
    return {"population_count": n, "labeled_count": labeled, "label_coverage": labeled / n if n else 0.0,
            "metrics": metrics, "breakdown": breakdown, "economics": economic, "transitions": transitions}


def native_economics_bindings(payload, kind, columns):
    """Translate an authenticated tool input without importing its calculator."""
    if payload is None:
        return None
    required = {"pd", "lgd", "utilization"} if kind == "limit" else {
        "ead", "pd", "lgd", "funding_rate", "term_months", "operating_cost_per_loan"} if kind == "pricing" else set()
    allowed = {name + suffix for name in required for suffix in ("_col", "_value")}
    if not isinstance(payload, dict) or not required or set(payload) - allowed:
        raise ValueError("unsupported native economics binding")
    result = {}
    for name in required:
        selected = [key for key in (name + "_col", name + "_value") if key in payload and payload[key] not in (None, "")]
        if len(selected) != 1:
            raise ValueError("complete unique native economics binding required")
        key, value = selected[0], payload[selected[0]]
        if key.endswith("_col"):
            if not isinstance(value, str) or value not in columns:
                raise ValueError("economics column absent from authenticated population")
            result[name] = {"column": value}
        elif not _numeric(value) or not math.isfinite(value):
            raise ValueError("finite native economic scalar required")
        else:
            result[name] = float(value)
    return result


def economics_reference(rows, values, kind, inputs, *, old=None):
    required = {"pd", "lgd", "utilization"} if kind == "limit" else {"ead", "pd", "lgd", "funding_rate", "term_months", "operating_cost_per_loan"}
    if set(inputs) != required:
        raise ValueError("complete explicit economics inputs required")
    def get(key, row):
        binding = inputs[key]
        value = row[binding["column"]] if isinstance(binding, dict) and set(binding) == {"column"} else binding
        if not _numeric(value) or not math.isfinite(value) or value < 0 or key in {"pd", "lgd", "utilization", "funding_rate"} and value > 1:
            raise ValueError("invalid economics value")
        if key == "term_months" and value <= 0:
            raise ValueError("positive economics term required")
        return float(value)
    if kind == "limit":
        ead, loss = [], []
        for row, value in zip(rows, values, strict=True):
            exposure = value * get("utilization", row)
            ead.append(exposure)
            loss.append(exposure * get("pd", row) * get("lgd", row))
        return {"expected_ead": math.fsum(ead), "expected_loss": math.fsum(loss)}
    by_row = []
    eads, rate_numerator = [], []
    for i, (row, rate) in enumerate(zip(rows, values, strict=True)):
        exposure, term = get("ead", row), get("term_months", row) / 12
        revenue, loss = exposure * rate * term, exposure * get("pd", row) * get("lgd", row)
        funding, operating = exposure * get("funding_rate", row) * term, get("operating_cost_per_loan", row)
        profit = revenue - loss - funding - operating
        delta = None if old is None else exposure * (rate - old[i]) * term
        by_row.append({"position": i, "revenue": revenue, "expected_loss": loss, "funding_cost": funding,
            "operating_cost": operating, "profit": profit, "roa": profit / exposure if exposure else None,
            "profit_delta_vs_baseline": delta})
        eads.append(exposure)
        rate_numerator.append(exposure * rate)
    total = math.fsum(eads)
    aggregate = {k: math.fsum(r[k] for r in by_row) for k in ("revenue", "expected_loss", "funding_cost", "operating_cost", "profit")}
    delta = None if old is None else math.fsum(r["profit_delta_vs_baseline"] for r in by_row)
    return {"total_ead": total, "ead_weighted_rate": math.fsum(rate_numerator) / total if total else None,
        **aggregate, "roa": aggregate["profit"] / total if total else None,
        "baseline_profit": None if delta is None else aggregate["profit"] - delta,
        "profit_delta_vs_baseline": delta, "by_row": by_row}


def stability_reference(development, observed, *, bucket_ids):
    if not development or not observed or any(v not in bucket_ids for v in [*development, *observed]):
        raise ValueError("exact fixed stability bucket population required")
    left, right = Counter(development), Counter(observed)
    expected = [left[v] / len(development) for v in bucket_ids]
    actual = [right[v] / len(observed) for v in bucket_ids]
    return {"development_count": len(development), "observed_count": len(observed),
            "development_distribution": expected, "observed_distribution": actual,
            "psi_vs_development": psi_reference(expected, actual)}
