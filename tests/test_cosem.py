from cellmap_models import cosem


def test_load_model():
    for model_name in cosem.model_names:
        model = cosem.load_model(model_name)
        assert model is not None
