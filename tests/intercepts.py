from collections import OrderedDict

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nsa import feature_map_shape_normalizers, intercepts, utils


class MLP3(nn.Module):
    def __init__(self, num_classes=10, d=100):
        super().__init__()
        self.lin1 = nn.Linear(28 * 28, d)
        self.d = d
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        return x

    def __str__(self):
        return "mlp3"


class CNN3(nn.Module):
    def __init__(self, num_classes=10, d=100):
        super().__init__()
        self.d = d
        self.conv1 = nn.Conv2d(1, d, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.avgpool1 = nn.AdaptiveAvgPool2d(18)
        self.conv2 = nn.Conv2d(d, d, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Outpu

        self.fc = nn.Linear(d, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.avgpool1(x)  # Output shape: (batch_size, d, 18, 18)
        x = self.conv2(x)
        x = self.act2(x)
        print("xxx", x.shape)  # Debugging line to check shape
        x = self.avgpool(x)  # Output shape: (batch_size, d, 1, 1)
        x = torch.flatten(x, 1)  # Flatten the output to (batch_size, d)
        x = self.fc(x)
        return x

    def __str__(self):
        return "cnn3"


@torch.no_grad()
@pytest.mark.parametrize(
    "model,shape_normalizer",
    [
        (MLP3(), feature_map_shape_normalizers.MLPFeatureMapShapeNormalizer()),
        (CNN3(), None),
    ],
)
def test_projection(model, shape_normalizer):
    rng = torch.Generator()
    rng.manual_seed(1)

    x = torch.randn((7, 1, 28, 28), generator=rng)

    U = utils.get_random_orthogonal_matrix(d=model.d, seed=1)
    expected = model(x).cpu().numpy()

    hook = None
    try:
        hook = model.act1.register_forward_hook(
            intercepts.construct_fh_with_projection(U, shape_normalizer=shape_normalizer)
        )

        actual = model(x).cpu().numpy()

        np.testing.assert_allclose(actual, expected, atol=1e-6)

    finally:
        if hook is not None:
            hook.remove()


def test_intercept():
    rng = torch.Generator()
    rng.manual_seed(1)

    d = 20
    x = torch.randn((2, 10), generator=rng)

    model = nn.Sequential(
        OrderedDict([
            ("lin1", nn.Linear(10, d)),
            ("act1", nn.ReLU()),
            ("lin2", nn.Linear(d, 5)),
        ])
    )

    with torch.no_grad():
        expected = nn.Sequential(*model[:2])(x).detach().cpu().numpy()

    hook = None
    try:
        hook = model.act1.register_forward_hook(intercepts.fh_intercept_output)

        x = x.clone()
        x.requires_grad = True

        model(x)

        actual = getattr(model.act1, intercepts.ATTRIBUTE_OUTPUT_KEY).detach().cpu().numpy()

        np.testing.assert_allclose(actual, expected, atol=1e-6)

    finally:
        if hook is not None:
            hook.remove()


@pytest.mark.parametrize(
    "model,inp,layer,expecteted_shape",
    [
        (MLP3(), torch.randn(20, 784), "act1", (20, 100)),
        (CNN3(), torch.randn(20, 1, 28, 28), "avgpool1", (20, 100, 18, 18)),
    ],
)
def test_feature_map_shape(model, inp, layer, expecteted_shape):
    b = inp.shape[0]

    dl = DataLoader(TensorDataset(inp), batch_size=b)

    actual = intercepts.get_feature_map_shape(
        model=model,
        layer=layer,
        dataloader=dl,
    )
    np.testing.assert_allclose(actual, expecteted_shape)
