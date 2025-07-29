import numpy as np
import pytest
import torch

from nsa.feature_map_shape_normalizers import (
    MLPFeatureMapShapeNormalizer,
    ViTFeatureMapShapeNormalizer,
)


@torch.no_grad()
@pytest.mark.parametrize(
    "shape, shape_normalizer",
    [
        ((10, 17, 768), ViTFeatureMapShapeNormalizer()),
        ((10, 32), MLPFeatureMapShapeNormalizer()),
    ],
)
def test_feature_map_transform(shape, shape_normalizer):
    x = torch.randn(*shape)

    # Apply the feature map transform
    out = shape_normalizer.to_original_shape(shape_normalizer.to_cnn_shape(x))

    np.testing.assert_allclose(out.numpy(), x.numpy(), rtol=1e-16, atol=1e-6)
