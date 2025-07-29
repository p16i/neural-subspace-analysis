import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nsa.evaluators import (
    ReconstructionErrorWithLowRankProjectionEvaluator,
)
from nsa.feature_map_shape_normalizers import ViTFeatureMapShapeNormalizer


class DummyModel(torch.nn.Module):
    def __init__(self, W=None, b=None):
        super().__init__()
        self.layer1 = nn.Identity()

    def forward(self, x):
        return self.layer1(x)


@pytest.mark.parametrize(
    "feature_map_shape,d,shape_normalizer",
    [
        ((8, 28, 28), 8, None),  # No shape normalizer
        ((17, 8), 8, ViTFeatureMapShapeNormalizer()),  # Identity shape normalizer
    ],
)
def test_reconstruction_error_with_low_rank_projection_evaluator(feature_map_shape, d, shape_normalizer):
    # Create dummy data: 10 samples, 4 features, 3 classes
    X = torch.randn(10, *feature_map_shape)  # 4 channels, 8x8 spatial dimensions
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=5)
    print(X.shape)

    # Dummy projection matrix: identity (no projection)
    U = torch.eye(d)
    arr_ks = np.array([1, 2, 4, 8])  # test for k=2 and k=4

    model = DummyModel()
    evaluator = ReconstructionErrorWithLowRankProjectionEvaluator()
    df = evaluator.evaluate(
        model,
        layer="layer1",
        dataloader=dataloader,
        U=U,
        arr_ks=arr_ks,
        device="cpu",
        verbose=False,
    )

    assert df.shape == (len(arr_ks), 4)  # 3 metrics

    assert df.columns.tolist() == ["k", *evaluator.metric_keys]

    assert np.all(df["norm"] >= 0)
    assert np.all(df["recon_err"] >= 0)

    np.testing.assert_equal(df["k"].values, arr_ks)  # Check if k values are as expected

    np.testing.assert_allclose(df["recon_err"].values[-1], 0)
