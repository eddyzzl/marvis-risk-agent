"""Conservative admission reservations shared by all model calls in one case.

The ceiling is the explicitly configured model context/output contract and a
frozen operator-supplied price book. This is not a provider billing guarantee.
Unknown usage keeps the entire reservation; it never creates free capacity.
"""

from __future__ import annotations

from decimal import Decimal


class AdmissionBudgetExceeded(ValueError):
    pass


class AggregateBudget:
    # Synchronization is owned by AttemptObserver's existing single lock.
    def __init__(self, *, context_window, max_total_tokens=None, max_cost=None,
                 currency=None, price=None, price_hash=None):
        self.context_window = context_window
        self.max_total_tokens = max_total_tokens
        self.max_cost = None if max_cost is None else Decimal(str(max_cost))
        self.currency, self.price_hash = currency, price_hash
        self.prompt_rate = Decimal(str(price.prompt_per_million)) if price else Decimal(0)
        self.completion_rate = Decimal(str(price.completion_per_million)) if price else Decimal(0)
        if self.max_cost is not None and (price is None or price.currency != currency):
            raise ValueError("cost budget requires a matching frozen price book")
        self.tokens = 0
        self.cost = Decimal(0)
        self.pending = {}
        self.unknown_attempts = 0
        self.violation = False

    @property
    def configured(self):
        return self.max_total_tokens is not None or self.max_cost is not None

    def _cost(self, prompt, completion):
        return (prompt * self.prompt_rate + completion * self.completion_rate) / Decimal(1_000_000)

    def policy(self):
        return {
            "max_total_tokens": self.max_total_tokens,
            "max_cost": None if self.max_cost is None else str(self.max_cost),
            "currency": self.currency,
            "price_book_sha256": self.price_hash,
            "prompt_reservation_per_attempt": self.context_window,
            "scope": "admission_under_declared_model_ceiling_and_supplied_prices",
            "provider_billing_guarantee": False,
            "reasoning_assumption": "completion usage and output ceiling include reasoning",
        }

    def reserve(self, ticket, output_cap):
        if not self.configured:
            return {}
        tokens = self.context_window + output_cap
        cost = self._cost(self.context_window, output_cap)
        if self.violation:
            raise AdmissionBudgetExceeded("provider_usage_exceeded_reserved_ceiling")
        if self.max_total_tokens is not None and self.tokens + tokens > self.max_total_tokens:
            raise AdmissionBudgetExceeded("aggregate_token_reservation_exhausted")
        if self.max_cost is not None and self.cost + cost > self.max_cost:
            raise AdmissionBudgetExceeded("aggregate_cost_reservation_exhausted")
        self.tokens += tokens
        self.cost += cost
        self.pending[ticket] = (output_cap, tokens, cost)
        return {"reserved_tokens": tokens, "reserved_cost": str(cost) if self.max_cost is not None else None}

    def settle(self, ticket, result):
        if ticket not in self.pending:
            return {}
        output_cap, tokens, cost = self.pending.pop(ticket)
        prompt, completion = result.get("prompt_tokens"), result.get("completion_tokens")
        reasoning = result.get("reasoning_tokens")
        def observed(field, value):
            maximum = result.get(f"observed_{field}_max")
            return max((v for v in (value, maximum) if type(v) is int and v >= 0), default=None)
        observed_prompt = observed("prompt_tokens", prompt)
        observed_completion = observed("completion_tokens", completion)
        observed_reasoning = observed("reasoning_tokens", reasoning)
        reasoning_conflict = observed_reasoning is not None and (
            observed_reasoning > output_cap or (type(completion) is int and observed_reasoning > completion)
        )
        if ((observed_prompt is not None and observed_prompt > self.context_window)
                or (observed_completion is not None and observed_completion > output_cap)
                or reasoning_conflict):
            self.violation = True
            charged_prompt = max(self.context_window, observed_prompt or 0)
            charged_completion = max(output_cap, observed_completion or 0, observed_reasoning or 0)
            self.tokens += charged_prompt + charged_completion - tokens
            self.cost += self._cost(charged_prompt, charged_completion) - cost
            return {"reservation_status": "provider_ceiling_violated", "aggregate_budget_violation": True}
        if not all(type(value) is int and value >= 0 for value in (prompt, completion)):
            self.unknown_attempts += 1
            return {"reservation_status": "retained_unknown_usage"}
        if result.get("usage_final") is not True:
            self.unknown_attempts += 1
            return {"reservation_status": "retained_unconfirmed_usage"}
        self.tokens -= tokens - prompt - completion
        self.cost -= cost - self._cost(prompt, completion)
        return {"reservation_status": "settled_reported_usage"}

    def snapshot(self):
        return {
            **self.policy(),
            "accounted_tokens_including_reservations": self.tokens,
            "accounted_cost_including_reservations": str(self.cost) if self.max_cost is not None else None,
            "pending_attempts": len(self.pending),
            "unknown_usage_attempts": self.unknown_attempts,
            "provider_ceiling_violated": self.violation,
        }
