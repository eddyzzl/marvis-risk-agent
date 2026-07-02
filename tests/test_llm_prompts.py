"""LLM-10: prompt registry invariants.

marvis.llm_prompts is the single source of truth for every system prompt's
text + version. These tests lock two things: (1) every prompt's text hash
matches its declared version, so an editor who changes wording without
bumping the version fails CI instead of silently drifting the eval baseline
(LLM-2) away from what the recorded prompt_version claims was live; (2) every
module that used to define its own SYS constant re-exports the exact same
text unchanged, so this refactor introduced zero prompt-wording changes.
"""

from __future__ import annotations

from marvis.llm_prompts import ALL_PROMPTS, PromptSpec, prompt_version_snapshot


# Hash-lock table: bump the version AND update the hash here in the same
# commit whenever a prompt's wording changes. A mismatch means either the
# text changed without a version bump (fix: bump the version) or the version
# was bumped without updating this lock (fix: update the hash below).
_LOCKED_HASHES = {
    "PLAN_SYS": (1, "b8f0c77f37f0eedd"),
    "REPLAN_SYS": (1, "872ddb4c5fe37492"),
    "EXPLORE_SYS": (1, "64a0439c1c1090a1"),
    "CRITIC_SYS": (1, "b9aef8096c81cc56"),
    "CLASSIFY_SYS": (1, "b74f7d825f9b1b10"),
    "GATE_SYSTEM_TEMPLATE": (1, "7ae1a3768ff2aaa6"),
    "GATE_INSTRUCTION_ROUTER_SYS": (1, "b55da0d0ac9ab4d9"),
    "AGENT_SYSTEM_PROMPT": (1, "8406896cbee9e955"),
    "WORD_CONCLUSION_SYSTEM_PROMPT": (1, "bc9e18c527e23e83"),
    "DISTILL_SYS": (1, "5d40eadf4107e9a8"),
    "AUTHOR_SYS": (1, "b89568bdc0ca1710"),
    "LEARN_SYS": (1, "40267be3b94b3147"),
    "CROSS_SYS": (1, "0d13fa241b855e51"),
    "REPORT_NARRATIVE_SYS": (1, "a6ff4690f78c4fe2"),
}


def test_every_registered_prompt_is_locked():
    registered_names = {spec.name for spec in ALL_PROMPTS}
    assert registered_names == set(_LOCKED_HASHES)


def test_prompt_text_hash_matches_locked_version():
    for spec in ALL_PROMPTS:
        expected_version, expected_hash = _LOCKED_HASHES[spec.name]
        assert spec.version == expected_version, (
            f"{spec.name}: version changed to {spec.version} without updating "
            "the lock table in tests/test_llm_prompts.py"
        )
        assert spec.text_hash == expected_hash, (
            f"{spec.name}: text changed but version is still {spec.version} — "
            "bump PromptSpec.version in marvis/llm_prompts.py and update the "
            "lock table here in the same commit."
        )


def test_prompt_names_are_unique():
    names = [spec.name for spec in ALL_PROMPTS]
    assert len(names) == len(set(names))


def test_version_tag_format():
    spec = PromptSpec(name="FOO_SYS", version=3, text="hello")
    assert spec.version_tag == "FOO_SYS_v3"


def test_prompt_version_snapshot_covers_all_prompts():
    snapshot = prompt_version_snapshot()
    assert snapshot == {spec.name: spec.version for spec in ALL_PROMPTS}
    assert snapshot["PLAN_SYS"] == 1


def test_call_site_constants_re_export_registry_text_unchanged():
    """Every call site's module-level SYS constant must equal the registry's
    text byte-for-byte — this refactor moved prompts into one module, it did
    not change any wording."""
    import marvis.agent.auto_drive as auto_drive
    import marvis.agent.instruction_router as instruction_router
    import marvis.agent.prompts as agent_prompts
    import marvis.agent_memory.distillation as distillation
    import marvis.drafts.authoring as authoring
    import marvis.drafts.learning as learning
    import marvis.feature.derive as derive
    import marvis.orchestrator.intent as intent
    import marvis.orchestrator.planner as planner
    import marvis.orchestrator.reviewer as reviewer
    import marvis.packs.modeling.tools as modeling_tools
    from marvis import llm_prompts as lp

    pairs = [
        (planner.PLAN_SYS, lp.PLAN_SYS.text),
        (planner.REPLAN_SYS, lp.REPLAN_SYS.text),
        (planner.EXPLORE_SYS, lp.EXPLORE_SYS.text),
        (reviewer.CRITIC_SYS, lp.CRITIC_SYS.text),
        (intent.CLASSIFY_SYS, lp.CLASSIFY_SYS.text),
        (auto_drive._SYSTEM_TEMPLATE, lp.GATE_SYSTEM_TEMPLATE.text),
        (instruction_router._SYSTEM, lp.GATE_INSTRUCTION_ROUTER_SYS.text),
        (distillation.DISTILL_SYS, lp.DISTILL_SYS.text),
        (authoring.AUTHOR_SYS, lp.AUTHOR_SYS.text),
        (learning.LEARN_SYS, lp.LEARN_SYS.text),
        (derive.CROSS_SYS, lp.CROSS_SYS.text),
        (agent_prompts.AGENT_SYSTEM_PROMPT, lp.AGENT_SYSTEM_PROMPT.text),
        (agent_prompts.WORD_CONCLUSION_SYSTEM_PROMPT, lp.WORD_CONCLUSION_SYSTEM_PROMPT.text),
        (modeling_tools.REPORT_NARRATIVE_SYS, lp.REPORT_NARRATIVE_SYS.text),
    ]
    for call_site_text, registry_text in pairs:
        assert call_site_text == registry_text
