"""Count committed plan revisions against the final native plan snapshots."""
from hashlib import sha256


def revision_observation(events, plans, *, journal_complete):
    kinds = ("structural_replan", "explore_append", "upstream_revision")
    final = {}
    if plans is not None:
        if not isinstance(plans, list) or len(plans) > 10000:
            raise ValueError("invalid_original_plan_snapshots")
        for row in plans:
            if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                    or not row["id"] or len(row["id"]) > 200
                    or type(row.get("replan_count")) is not int or row["replan_count"] < 0):
                raise ValueError("invalid_original_plan_snapshot")
            identity = sha256(row["id"].encode()).hexdigest()
            if identity in final:
                raise ValueError("duplicate_original_plan_snapshot")
            final[identity] = row["replan_count"]
    grouped = {}
    for event in events:
        if event.get("event") == "plan_revision":
            grouped.setdefault(event["identity"], []).append(event)
    observed = []
    total = dict.fromkeys(kinds, 0)
    for identity in sorted(set(grouped) | set(final)):
        rows = grouped.get(identity, [])
        initial = rows[0]["revision"] if rows and rows[0]["kind"] == "created" else None
        valid = initial is not None
        current = initial
        counts = dict.fromkeys(kinds, 0)
        for event in rows[1:] if initial is not None else rows:
            kind = event["kind"]
            if kind in counts:
                counts[kind] += 1
            valid &= kind in counts and current is not None and event["revision"] == current + 1
            current = event["revision"]
        complete = bool(valid and identity in final and final[identity] == current)
        observed.append({"identity": identity, "initial_revision": initial,
                         "final_revision": final.get(identity), "known_counts": counts,
                         "complete": complete})
        for kind, count in counts.items():
            total[kind] += count
    return {
        "scope": "committed_in_case_revisions_separate_from_planner_attempts",
        "complete": bool(journal_complete and plans is not None and all(row["complete"] for row in observed)),
        "snapshot_available": plans is not None,
        "known_counts": total,
        "plan_denominator": len(observed),
        "unknown_plans": sum(not row["complete"] for row in observed),
        "plans": observed,
    }
