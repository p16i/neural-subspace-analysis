import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nsa import estimators


@pytest.mark.parametrize(
    "N,d,bs",
    [
        (100, 45, 7),
        (100, 45, 100),
        (100, 45, 100),
    ],
)
def test_estimate_cov_2d_tensor(N, d, bs):
    torch.manual_seed(1)

    X = torch.randint(0, 10, size=(N, d))

    dl = DataLoader(TensorDataset(X), batch_size=bs)

    expected = ((X.T @ X) / N).detach().numpy()

    cov_estimator = estimators.CovarianceEstimator()

    for (x,) in dl:
        cov_estimator.update(x)

    actual = cov_estimator.compute()

    np.testing.assert_allclose(actual, expected, atol=1e-3)


@pytest.mark.parametrize(
    "N,d,h,w,bs",
    [
        (100, 45, 8, 6, 7),
        (100, 32, 8, 6, 50),
    ],
)
def test_estimate_cov_4d_tensor(N, d, h, w, bs):
    torch.manual_seed(1)

    X = torch.randint(0, 10, size=(N, d, h, w))

    dl = DataLoader(TensorDataset(X), batch_size=bs)

    flatten_x = X.permute(1, 0, 2, 3).flatten(start_dim=1).T

    expected = ((flatten_x.T @ flatten_x) / (N * h * w)).detach().numpy()

    cov_estimator = estimators.CovarianceEstimator()

    for (x,) in dl:
        cov_estimator.update(x)

    actual = cov_estimator.compute()

    np.testing.assert_allclose(actual, expected, atol=1e-3)
