from marvis.validation.lift_ranking import (
    assess_bin_rows,
    assess_lift_ranking,
    head_lift_strength,
    tail_lift_strength,
)


def test_head_lift_just_below_one_is_weak_not_pass():
    assert head_lift_strength(0.99) == "weak"
    assert head_lift_strength(0.79) == "ok"
    assert head_lift_strength(0.40) == "strong"
    assert head_lift_strength(1.02) == "fail"


def test_tail_lift_just_above_one_is_weak():
    assert tail_lift_strength(1.05) == "weak"
    assert tail_lift_strength(1.80) == "ok"
    assert tail_lift_strength(2.40) == "strong"
    assert tail_lift_strength(0.95) == "fail"


def test_bin_rows_detect_monotonic_and_broken_ranking():
    monotonic = [
        {"lift": 0.20, "bad_rate": 0.01},
        {"lift": 0.80, "bad_rate": 0.04},
        {"lift": 2.50, "bad_rate": 0.12},
    ]
    broken = [
        {"lift": 0.90, "bad_rate": 0.08},
        {"lift": 0.40, "bad_rate": 0.02},
        {"lift": 0.30, "bad_rate": 0.01},
        {"lift": 1.10, "bad_rate": 0.09},
    ]

    good = assess_bin_rows(monotonic, source="train_aligned")
    bad = assess_bin_rows(broken, source="train_aligned")

    assert good["bad_rate_monotonicity"] == "monotonic"
    assert good["head_group_strength"] == "strong"
    assert good["tail_group_strength"] == "strong"
    assert bad["bad_rate_monotonicity"] == "broken"
    assert "head_lift_near_one" in bad["notes"]


def test_assess_lift_ranking_uses_overall_5pct_and_bin_tables():
    assessment = assess_lift_ranking(
        {
            "overall": [
                {
                    "split": "oot",
                    "head_lift_5pct": 0.99,
                    "tail_lift_5pct": 1.04,
                }
            ],
            "bin_tables": {
                "oot": [
                    {"lift": 0.99, "bad_rate": 0.07},
                    {"lift": 1.00, "bad_rate": 0.08},
                    {"lift": 1.02, "bad_rate": 0.09},
                ]
            },
        }
    )
    oot = assessment["splits"][2]
    assert oot["split"] == "oot"
    assert oot["head_5pct_strength"] == "weak"
    assert oot["tail_5pct_strength"] == "weak"
    assert "head_lift_near_one" in oot["train_aligned"]["notes"]
    assert "tail_lift_near_one" in oot["train_aligned"]["notes"]
