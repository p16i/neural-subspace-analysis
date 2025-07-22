import pytest

import numpy as np

import torch
from nsa import utils


@pytest.mark.parametrize(
    "start,stop,step,expected",
    [
        (2, 5, 1, [2, 3, 4, 5]),
        (2, 5, 2, [2, 4, 5]),
    ],
)
def test(start, stop, step, expected):

    np.testing.assert_equal(utils.arange_with_grid(start, stop, step=step), expected)


@torch.no_grad()
def test_reshape_tensor():
    x = torch.randn(10, 8)

    expected = x.unsqueeze(2).unsqueeze(3).numpy()
    actual = utils.reshape_tensor_to_cnn_like(x).numpy()
    np.testing.assert_allclose(actual, expected)


def test_flatten_4d_tensor():
    torch.manual_seed(1)

    N = 10
    h = w = 5
    d = 32

    x = torch.rand(size=(N, d, h, w))

    actual = utils.flatten_4d_tensor(x)

    assert actual.shape == (N * h * w, d)

    for i in range(N):
        for j in range(h):
            for k in range(w):
                expected = x[i, :, j, k]
                pos = i * (h * w) + j * w + k
                np.testing.assert_allclose(actual[pos, :], expected)


@pytest.mark.parametrize(
    "txt,default_layers,expected",
    [
        ("layer2,@3", ["layer1", "layer2", "layer3", "layer4"], ["layer2", "layer4"]),
        ("@0,@2", ["layer1", "layer2", "layer3", "layer4"], ["layer1", "layer3"]),
        (
            "layer3,layer4",
            ["layer1", "layer2", "layer3", "layer4"],
            ["layer3", "layer4"],
        ),
        (
            "@0,@1,@2,@3",
            ["layer1", "layer2", "layer3", "layer4"],
            ["layer1", "layer2", "layer3", "layer4"],
        ),
    ],
)
def test_parse_layers(txt, default_layers, expected):
    actual = utils.parse_layers(txt, default_layers)

    np.testing.assert_equal(actual, expected)
