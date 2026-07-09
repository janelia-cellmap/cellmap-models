import torch

from cellmap_models.model_export import (
    ModelMetadata,
    parse_activation_to_torch,
    parse_normalization_min_max,
)


def test_formal_preprocessing_metadata_parses():
    metadata = ModelMetadata(
        input_normalization={"method": "min_max", "min": -1.0, "max": 1.0},
        inference_output_activation={"name": "sigmoid", "apply_after_model": True},
    )

    assert metadata.input_normalization.model_dump(exclude_none=True) == {
        "method": "min_max",
        "min": -1.0,
        "max": 1.0,
    }
    assert parse_normalization_min_max(metadata.input_normalization) == (-1.0, 1.0)
    activation = parse_activation_to_torch(metadata.inference_output_activation)
    assert torch.allclose(activation(torch.tensor([0.0])), torch.tensor([0.5]))


def test_activation_parser_avoids_double_sigmoid_when_included_in_model():
    metadata = ModelMetadata(
        inference_output_activation={"name": "sigmoid", "apply_after_model": False},
    )

    activation = parse_activation_to_torch(metadata.inference_output_activation)
    value = torch.tensor([0.25])
    assert torch.equal(activation(value), value)


def test_legacy_string_metadata_still_parses():
    metadata = ModelMetadata(
        input_normalization="normalize intensities to [-1, 1]",
        inference_output_activation="sigmoid applied after model logits",
    )

    assert parse_normalization_min_max(metadata.input_normalization) == (-1.0, 1.0)
    assert metadata.inference_output_activation.name == "sigmoid"
    assert metadata.inference_output_activation.apply_after_model is True
