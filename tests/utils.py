import numpy as np
import pytest
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


@torch.no_grad()
def test_solve_eigh_basic():
    # Symmetric matrix
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    eigvals, eigvecs = utils.eigh(torch.from_numpy(A))

    eigvals = eigvals.numpy()
    eigvecs = eigvecs.numpy()
    # Check eigenvalues
    expected_eigvals = np.linalg.eigvalsh(A)
    np.testing.assert_allclose(eigvals, np.sort(expected_eigvals)[::-1], rtol=1e-6)
    # Check eigenvectors (orthonormal)
    assert np.allclose(np.dot(eigvecs.T, eigvecs), np.eye(eigvecs.shape[1]), atol=1e-6)
    # Check reconstruction
    A_recon = eigvecs @ np.diag(eigvals) @ eigvecs.T
    np.testing.assert_allclose(A, A_recon, rtol=1e-6)
    assert (
        eigvals[:-1] - eigvals[1:] >= 0
    ).all(), "Eigenvalues should be sorted in ascending order"


def test_solve_eigh_identity():
    A = np.eye(4)
    eigvals, eigvecs = utils.eigh(torch.from_numpy(A))
    np.testing.assert_allclose(eigvals, np.ones(4))
    np.testing.assert_allclose(eigvecs @ eigvecs.T, np.eye(4), atol=1e-6)


def test_solve_eigh_diag():
    eigvals = [1, 4.0, 3, 2.0]
    A = np.diag(eigvals)
    eigvals, eigvecs = utils.eigh(torch.from_numpy(A))
    np.testing.assert_allclose(eigvals, sorted(eigvals)[::-1])
    np.testing.assert_allclose(eigvecs @ eigvecs.T, np.eye(4), atol=1e-6)


def test_solve_eigh_random_symmetric():
    np.random.seed(42)
    A = np.random.randn(5, 5)
    A = (A + A.T) / 2  # Make symmetric
    eigvals, eigvecs = utils.eigh(torch.from_numpy(A))

    eigvals = eigvals.numpy()
    eigvecs = eigvecs.numpy()

    # Eigenvectors should be orthonormal
    np.testing.assert_allclose(eigvecs.T @ eigvecs, np.eye(5), atol=1e-6)
    # Reconstruction
    A_recon = eigvecs @ np.diag(eigvals) @ eigvecs.T
    np.testing.assert_allclose(A, A_recon, rtol=1e-6)
