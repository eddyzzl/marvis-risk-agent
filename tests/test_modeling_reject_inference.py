import pytest

import marvis.packs.modeling as modeling
from marvis.packs.modeling.recipes import list_recipes
from marvis.packs.modeling.reject_inference import reject_inference


def test_reject_inference_stub_requires_methodology_review():
    with pytest.raises(NotImplementedError) as excinfo:
        reject_inference(method="heckman")

    message = str(excinfo.value)
    assert "methodology review" in message
    assert "Heckman" in message
    assert "parceling" in message
    assert "fuzzy augmentation" in message


def test_reject_inference_is_not_registered_as_available_modeling_capability():
    assert "reject_inference" not in modeling.__all__
    assert all(recipe.id != "reject_inference" for recipe in list_recipes())
