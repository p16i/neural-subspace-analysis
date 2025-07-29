import typing
from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from subana import feature_map_shape_normalizers, intercepts, utils


def estimate_cov_mat_at_layer(
    model: nn.Module,
    layer: str,
    dataloader: DataLoader,
    device="cpu",
    pre_transform: typing.Optional[typing.Callable] = None,
) -> torch.Tensor:
    """Estimate (uncentered) covariance matrix at a given layer.

    Args:
        model (nn.Module): model to estimate covariance matrix for.
        layer (str): at which layer we want to extract the covariance matrix
        dataloader (DataLoader): use the first item in each batch as input to the model.
        device (str, optional): _description_. Defaults to "cpu".
        pre_transform (typing.Optional[typing.Callable], optional): _description_. Defaults to None.

    Raises:
        e: _description_

    Returns:
        torch.Tensor: _description_
    """

    estimator = CovarianceEstimator()

    layer, shape_normalizer = feature_map_shape_normalizers.resolve_shape_normalizer(
        model=model,
        layer=layer,
        dataloader=dataloader,
        device=device,
    )

    hook = None

    try:
        module = intercepts.get_module_for_layer(model=model, layer=layer)
        hook = module.register_forward_hook(intercepts.fh_intercept_output)

        for batch in tqdm(dataloader, desc=f"[layer={layer}] estimating covariance matrix"):
            x = batch[0] if isinstance(batch, Sequence) else batch

            x = x.to(device)
            _ = model(x)

            layer_output = intercepts.get_module_output(module)

            if pre_transform is not None:
                layer_output = pre_transform(layer_output)

            if shape_normalizer is not None:
                layer_output = shape_normalizer.to_cnn_shape(layer_output)

            estimator.update(layer_output)
    finally:
        if hook is not None:
            hook.remove()

    cov_mat = estimator.compute()
    return cov_mat


class CovarianceEstimator:
    """Batch-wisee covariance matrix estimator."""

    def __init__(self):
        self._cov_mat = None
        self._N = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if len(x.shape) not in [2, 4]:
            raise ValueError(f"Expected input tensor with 2 or 4 dimensions, got shape {x.shape}")  # noqa: TRY003

        if len(x.shape) == 4:
            x = utils.flatten_4d_tensor(x)

        N, _ = x.shape

        # remark: we use cpu here to avoid memory issues when d is large.
        curr_mat = ((x.T @ x) / N).cpu()

        if self._cov_mat is None:
            self._cov_mat = curr_mat
            self._N = N
        else:
            self._cov_mat = ((self.N * self._cov_mat) + (N * curr_mat)) / (self.N + N)
            self._N = N + self.N

    @property
    def cov_mat(self) -> typing.Union[torch.Tensor, None]:
        return self._cov_mat

    @property
    def N(self) -> int:
        return self._N

    def compute(self):
        if self.cov_mat is None:
            raise ValueError("Covariance matrix has not been computed yet.")  # noqa: TRY003

        return self.cov_mat.detach()
