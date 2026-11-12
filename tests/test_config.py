import pytest
from nanotransformer.config import TransformerConfig


def test_transformer_config_validation_and_derived_properties():
    cfg = TransformerConfig(dim=128, n_heads=8, n_kv_heads=2)
    assert cfg.head_dim == 16
    assert cfg.kv_groups == 4

    with pytest.raises(ValueError, match="divisible by n_heads"):
        TransformerConfig(dim=100, n_heads=8, n_kv_heads=2)

    with pytest.raises(ValueError, match="divisible by n_kv_heads"):
        TransformerConfig(dim=128, n_heads=8, n_kv_heads=3)
