from __future__ import annotations

import importlib
from importlib import resources
from pathlib import Path
import subprocess
import tomllib


def _pyproject() -> dict:
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def _requirement_names(requirements: list[str]) -> set[str]:
    names: set[str] = set()
    for requirement in requirements:
        name = requirement.split("[", 1)[0].split("<", 1)[0].split(">", 1)[0]
        name = name.split("=", 1)[0].strip().lower().replace("_", "-")
        names.add(name)
    return names


def test_project_identity_uses_marvis_package_and_scripts():
    pyproject = _pyproject()

    assert pyproject["project"]["name"] == "marvis"
    assert pyproject["project"]["scripts"] == {
        "marvis": "marvis.__main__:main",
        "marvis-risk-agent": "marvis.__main__:main",
    }
    assert "marvis" in pyproject["tool"]["setuptools"]["package-data"]
    assert "riskmodel" + "_checker" not in pyproject["tool"]["setuptools"]["package-data"]

    package = importlib.import_module("marvis")
    assert package.__name__ == "marvis"
    assert resources.files("marvis").joinpath("static/app.js").is_file()


def test_pyproject_includes_notebook_data_science_runtime_stack():
    dependency_names = _requirement_names(_pyproject()["project"]["dependencies"])

    expected = {
        "fastapi",
        "uvicorn",
        "nbformat",
        "nbclient",
        "ipykernel",
        "ipython",
        "jupyter-client",
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "joblib",
        "matplotlib",
        "seaborn",
        "openpyxl",
        "pyarrow",
        "xlrd",
        "python-docx",
        "pydantic",
        "sklearn2pmml",
        "pypmml",
        "xgboost",
        "lightgbm",
        "catboost",
    }

    assert expected <= dependency_names


def test_tracked_files_do_not_reference_legacy_identity():
    raw_tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    tracked = [
        item.decode("utf-8")
        for item in raw_tracked.split(b"\0")
        if item
    ]
    offenders: list[str] = []
    banned = ("riskmodel" + "_checker", "riskmodel" + "-checker")

    for relative_path in tracked:
        path = Path(relative_path)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(token in text for token in banned) or any(
            token in relative_path for token in banned
        ):
            offenders.append(relative_path)

    assert offenders == []
