from __future__ import annotations


def reject_inference(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError(
        "reject inference requires methodology review before implementation; "
        "see blueprint 15.1. Candidate methods: Heckman / parceling / "
        "augmentation / fuzzy augmentation."
    )


__all__ = ["reject_inference"]
