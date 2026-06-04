# word_conclusion_writer

Use this skill only for Agent-mode final Word conclusion drafts.

Input evidence must come from structured platform outputs: task metadata, scan result, Notebook/reproducibility evidence, validation results, prior Agent analysis messages, and the reference-writing summary in the Agent P1 spec.

Output a JSON object with exactly these keys:

```json
{
  "TEXT:pressure_test_summary": "...",
  "TEXT:pressure_impact_recommendation": "...",
  "TEXT:final_validation_conclusion": "..."
}
```

Rules:

- Do not invent metrics, thresholds, files, or regulatory conclusions that are not present in the evidence.
- `TEXT:pressure_test_summary` explains test purpose, method, and observed pressure-risk pattern, and must summarize high-risk, medium-risk, and low-risk data sources when evidence is available.
- `TEXT:pressure_impact_recommendation` gives operational recommendations by pressure impact level, including monitoring, substitution, fallback, or launch restrictions for the risk tiers identified above.
- `TEXT:final_validation_conclusion` should be slightly longer than the other two fields: 1 to 2 paragraphs, roughly 120 to 220 Chinese characters or more when evidence requires it. It should connect development process, Notebook reproducibility, score consistency, discrimination, stability, pressure-test findings, report-output state, and a final prudent judgment.
- If evidence is missing, say it is not yet possible to judge and recommend the exact review step instead of filling positive boilerplate.
