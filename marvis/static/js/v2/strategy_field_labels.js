/** Display metadata only: option values and submitted physical fields stay exact. */

export function applyStrategyFieldLabels(root, payload) {
  if (!root) return;
  const preference = root.querySelector?.('[data-candidate-lab-workflow="strategy_workbench_preferences"] [data-candidate-lab-field="field_display_mode"]');
  const mode = ["physical", "business", "both"].includes(preference?.value) ? preference.value : "both";
  const fields = payload?.field_labels?.fields;
  const labels = new Map(Array.isArray(fields) ? fields.map((field) => [field.physical, field.business_name]) : []);
  const fieldSelectors = new Set(["cross_search_features", "cross_rule_features", "interactive_tree_feature"]);
  for (const option of root.querySelectorAll?.("select option") || []) {
    const field = option.closest?.("select")?.dataset?.candidateLabField;
    if (!fieldSelectors.has(field) || !labels.has(option.value)) continue;
    // textContent renders a label, never HTML, and cannot change submitted values.
    const physical = option.value;
    const business = labels.get(physical);
    if (option.dataset && !Object.hasOwn(option.dataset, "fieldLabelSuffix")) {
      option.dataset.fieldLabelSuffix = option.textContent.startsWith(physical) ? option.textContent.slice(physical.length) : "";
    }
    const displayed = mode === "physical" || !business ? physical
      : mode === "business" ? business : `${business} (${physical})`;
    option.textContent = displayed + (option.dataset?.fieldLabelSuffix || "");
    option.title = business ? `${physical} · ${business}` : physical;
  }
  // Free-text field controls offer canonical choices with business labels. The
  // submitted value stays physical; arbitrary prose is never an evidence pointer.
  if (root.ownerDocument && Array.isArray(fields)) {
    let list = root.querySelector('[data-strategy-field-label-options]');
    if (!list) {
      list = root.ownerDocument.createElement("datalist");
      list.id = "strategyConfirmedFieldLabelOptions";
      list.dataset.strategyFieldLabelOptions = "1";
      root.appendChild(list);
    }
    list.replaceChildren(...fields.map((field) => {
      const option = root.ownerDocument.createElement("option");
      option.value = field.physical;
      option.label = mode === "physical" || !field.business_name ? field.physical
        : mode === "business" ? field.business_name : `${field.business_name} (${field.physical})`;
      return option;
    }));
    for (const input of root.querySelectorAll('input[data-candidate-lab-field]')) {
      if (!input.readOnly && /(?:column|_col|_feature|_field)$/.test(input.dataset.candidateLabField)) {
        input.setAttribute("list", list.id);
      }
    }
  }
}
