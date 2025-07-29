import typing

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from nsa import estimators


class Case:
    input_shape: typing.Tuple[int, ...] = ()

    def groundtruth(self, x: torch.Tensor) -> typing.Tuple[int, torch.Tensor]:
        pass

    def construct_additional_transform(
        self,
    ) -> typing.Optional[typing.Callable[..., torch.Tensor]]:
        pass


class CaseMLP(Case):
    input_shape = (10, 64)

    def groundtruth(self, x: torch.Tensor) -> typing.Tuple[int, torch.Tensor]:
        # For MLP, we assume the input is flattened
        assert len(x.shape) == 2, f"Expected 2D tensor, got {x.shape}"

        n, d = x.shape

        cov = (x.T @ x) / n
        return d, cov

    def construct_additional_transform(
        self,
    ):
        return None


class CaseCNN(Case):
    input_shape = (10, 32, 8, 8)

    def groundtruth(self, x: torch.Tensor) -> typing.Tuple[int, torch.Tensor]:
        # For CNN, flatten spatial dimensions and channels
        assert len(x.shape) == 4, f"Expected 4D tensor, got {x.shape}"
        n, c, h, w = x.shape
        # Flatten each sample to (n, c*h*w)
        x_flat = x.permute(0, 2, 3, 1).reshape(n * h * w, c)
        cov = (x_flat.T @ x_flat) / (n * h * w)
        return c, cov

    def construct_additional_transform(self):
        # Flatten spatial dimensions and channels for covariance computation
        def transform(x):
            n, c, h, w = x.shape
            return x.permute(0, 2, 3, 1).reshape(n * h * w, c)

        return transform


class CaseViT(Case):
    input_shape = (10, 16 + 1, 64)  # (batch, num_patches + [cls], embed_dim)

    def groundtruth(self, x: torch.Tensor) -> typing.Tuple[int, torch.Tensor]:
        # x: (n, num_patches, embed_dim)
        assert len(x.shape) == 3, f"Expected 3D tensor, got {x.shape}"
        n, num_patches, embed_dim = x.shape
        # Flatten all patches into a single batch
        x_flat = x.reshape(n * num_patches, embed_dim)
        cov = (x_flat.T @ x_flat) / (n * num_patches)
        return embed_dim, cov

    def construct_additional_transform(self):
        # Flatten all patches into a single batch for covariance computation
        def transform(x):
            n, num_patches, embed_dim = x.shape
            return x.reshape(n * num_patches, embed_dim)

        return transform


class CaseViTCLSOnly(Case):
    input_shape = (10, 16 + 1, 64)  # (batch, num_patches + [cls], embed_dim)

    def groundtruth(self, x: torch.Tensor) -> typing.Tuple[int, torch.Tensor]:
        # x: (n, num_patches, embed_dim)
        assert len(x.shape) == 3, f"Expected 3D tensor, got {x.shape}"
        n, num_patches, embed_dim = x.shape
        # Assume [cls] token is at position 0 for each sample
        x_cls = x[:, 0, :]  # shape: (n, embed_dim)
        cov = (x_cls.T @ x_cls) / n
        return embed_dim, cov

    def construct_additional_transform(self):
        # Extract only the [cls] token for each sample
        def transform(x):
            # x: (n, num_patches, embed_dim)
            return x[:, 0, :]  # (n, embed_dim)

        return transform


@pytest.mark.parametrize(
    "case",
    [CaseMLP(), CaseCNN(), CaseViT(), CaseViTCLSOnly()],
)
@torch.no_grad()
def test(case: Case):

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Identity()

        def forward(self, x):
            return self.layer1(x)

    torch.manual_seed(1)

    # simulate CNN feature map
    x = torch.randn(*case.input_shape)
    dl = torch.utils.data.DataLoader(TensorDataset(x), batch_size=10, shuffle=False)

    cov = estimators.estimate_cov_mat_at_layer(
        model=Model(),
        layer="layer1",
        dataloader=dl,
        transform=case.construct_additional_transform(),
        device="cpu",
    )

    d, expected_cov = case.groundtruth(x)

    assert cov is not None
    assert cov.shape == (d, d), "Covariance matrix shape mismatch"

    np.testing.assert_allclose(
        cov.numpy(),
        expected_cov.numpy(),
        atol=1e-3,
        err_msg="Covariance matrix does not match expected values",
    )
