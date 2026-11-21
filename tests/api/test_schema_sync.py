from typing import Literal, get_args, get_origin

from credit_mlops.api.schemas import Application
from credit_mlops.schema import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


def test_application_matches_schema():
    fields = Application.model_fields
    assert list(fields) == FEATURES
    for name, spec in NUMERIC_FEATURES.items():
        field = fields[name]
        assert field.annotation is (int if spec.integer else float), name
        limits = {
            key: getattr(m, key) for m in field.metadata for key in ("ge", "le") if hasattr(m, key)
        }
        assert limits == {"ge": spec.min, "le": spec.max}, name
    for name, categories in CATEGORICAL_FEATURES.items():
        annotation = fields[name].annotation
        assert get_origin(annotation) is Literal and get_args(annotation) == categories, name
