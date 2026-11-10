"""Home batch validation (2026-11-10): materials dropped on the home page are
grouped into models by folder or by file-name stem, so several models start one
validation batch from the 模型验证 card. Grouping uses evidence only; anything
it cannot place stays unassigned for the user.
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = (ROOT / "marvis" / "static" / "js" / "home-validation-models.js").as_uri()


def _run(script: str) -> dict:
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", f"const m = await import({json.dumps(MODULE)});\n{script}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout)


_HELPERS = """
const f = (name, relativePath) => ({ file: { name, size: 1 }, relativePath: relativePath || name });
const view = (grouped) => grouped && {
  models: grouped.models.map((model) => [model.name, model.items.map((item) => item.file.name), m.modelMaterialGaps(model.items)]),
  unassigned: grouped.unassigned.map((item) => item.file.name),
};
"""


def test_models_group_by_file_name_stem_and_share_one_dictionary():
    data = _run(_HELPERS + """
process.stdout.write(JSON.stringify(view(m.groupValidationModels([
  f("A卡_notebook.ipynb"), f("A卡_model.pmml"), f("A卡_sample.csv"),
  f("B卡.ipynb"), f("B卡.pmml"), f("B卡_样本.parquet"), f("数据字典.xlsx"),
]))));
""")
    assert data == {
        "models": [
            ["A卡", ["A卡_notebook.ipynb", "A卡_model.pmml", "A卡_sample.csv", "数据字典.xlsx"], []],
            ["B卡", ["B卡.ipynb", "B卡.pmml", "B卡_样本.parquet", "数据字典.xlsx"], []],
        ],
        "unassigned": [],
    }


def test_models_group_by_sub_folder_when_a_folder_is_dropped():
    data = _run(_HELPERS + """
process.stdout.write(JSON.stringify(view(m.groupValidationModels([
  f("a.ipynb", "batch/A/a.ipynb"), f("m.pmml", "batch/A/m.pmml"), f("s.csv", "batch/A/s.csv"),
  f("x.ipynb", "batch/B/x.ipynb"), f("y.pmml", "batch/B/y.pmml"), f("z.csv", "batch/B/z.csv"),
  f("dict.xlsx", "batch/dict.xlsx"),
]))));
""")
    assert [model[0] for model in data["models"]] == ["A", "B"]
    assert all(model[2] == [] for model in data["models"])
    assert data["unassigned"] == []


def test_one_model_stays_a_single_model_and_unmatched_tables_stay_unassigned():
    data = _run(_HELPERS + """
const single = m.groupValidationModels([f("a.ipynb"), f("a.pmml"), f("s.csv"), f("d_dict.csv")]);
const ambiguous = view(m.groupValidationModels([f("a.ipynb"), f("a.pmml"), f("b.ipynb"), f("b.pmml"), f("s1.csv"), f("s2.csv")]));
process.stdout.write(JSON.stringify({ single, ambiguous }));
""")
    assert data["single"] is None
    # Two samples that name neither model are not guessed into either.
    assert data["ambiguous"]["unassigned"] == ["s1.csv", "s2.csv"]
    assert [model[2] for model in data["ambiguous"]["models"]] == [["sample", "dictionary"], ["sample", "dictionary"]]


def test_models_are_ready_only_when_named_complete_and_within_the_batch_limit():
    data = _run(_HELPERS + """
const full = (key) => [f(key + ".ipynb"), f(key + ".pmml"), f(key + "_sample.csv"), f(key + "_dict.csv")];
process.stdout.write(JSON.stringify({
  ready: m.validationModelsReady([{ name: "A", items: full("a") }, { name: "B", items: full("b") }]),
  unnamed: m.validationModelsReady([{ name: "A", items: full("a") }, { name: " ", items: full("b") }]),
  incomplete: m.validationModelsReady([{ name: "A", items: full("a") }, { name: "B", items: [f("b.ipynb")] }]),
  one: m.validationModelsReady([{ name: "A", items: full("a") }]),
  eleven: m.validationModelsReady(Array.from({ length: 11 }, (_, index) => ({ name: "M" + index, items: full("m" + index) }))),
  gaps: m.modelMaterialGaps([f("b.ipynb")]),
}));
""")
    assert data == {
        "ready": True,
        "unnamed": False,
        "incomplete": False,
        "one": False,
        "eleven": False,
        "gaps": ["sample", "pmml", "dictionary"],
    }
