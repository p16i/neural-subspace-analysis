import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nsa.evaluators.accuracy import AccuracyWithLowRankProjectionEvaluator
from nsa.feature_map_shape_normalizers import ViTFeatureMapShapeNormalizer


class DummyCNN(torch.nn.Module):
    def __init__(self, W=None, b=None):
        super().__init__()
        self.layer1 = nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Dummy pooling layer
        self.fc = nn.Linear(8, 3)  # Dummy linear layer for classification

    def forward(self, x):
        out = self.layer1(x)
        out = self.avgpool(out)  # Apply the pooling layer
        out = out.view(out.size(0), -1)  # Flatten the output
        out = self.fc(out)  # Apply the linear layer
        return out


class DummyViT(torch.nn.Module):
    def __init__(self, W=None, b=None):
        super().__init__()
        self.layer1 = nn.Identity()
        self.fc = nn.Linear(8, 3)  # Dummy linear layer for classification

    def forward(self, x):
        out = self.layer1(x)
        # take [cls] token and sum over the sequence length
        out = out[:, 0, :]

        out = self.fc(out)  # Apply the linear layer

        return out


@pytest.mark.parametrize(
    "model,feature_map_shape,d,shape_normalizer",
    [
        (DummyCNN(), (8, 28, 28), 8, None),  # No shape normalizer
        (
            DummyViT(),
            (17, 8),
            8,
            ViTFeatureMapShapeNormalizer(),
        ),  # Identity shape normalizer
    ],
)
def test_accuracy_with_low_rank_projection_evaluator(
    model, feature_map_shape, d, shape_normalizer
):
    # Create dummy data: 10 samples, 4 features, 3 classes
    X = torch.randn(10, *feature_map_shape)  # 4 channels, 8x8 spatial dimensions
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=5)

    # Dummy projection matrix: identity (no projection)
    U = torch.eye(d)
    arr_ks = [2, 4]  # test for k=2 and k=4

    evaluator = AccuracyWithLowRankProjectionEvaluator(num_classes=3)

    df = evaluator.evaluate(
        model,
        layer="layer1",
        dataloader=dataloader,
        U=U,
        arr_ks=np.array(arr_ks),
        device="cpu",
        verbose=False,
    )

    assert df.shape == (2, 3)  # 2 ks, 3 metrics

    assert df.columns.tolist() == ["k", *evaluator.metric_keys]

    assert np.all(df["acc"] >= 0)
    assert np.all(df["xent"] >= 0)

    np.testing.assert_equal(df["k"].values, arr_ks)  # Check if k values are as expected
