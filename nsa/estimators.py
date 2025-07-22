import typing
import torch

from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from nsa import utils, intercepts


def estimate_cov_mat_at_layer(
    model: nn.Module, layer: str, dataloader: DataLoader, device="cpu"
) -> torch.Tensor:

    estimator = CovarianceEstimator()

    hook = None

    try:
        module = intercepts.get_module_for_layer(model=model, layer=layer)
        hook = module.register_forward_hook(intercepts.fh_intercept_output)

        for x, y in tqdm(
            dataloader, desc=f"[layer={layer}] estimating covariance matrix"
        ):
            x = x.to(device)
            _ = model(x)

            layer_output = intercepts.get_module_output(module)

            estimator.update(layer_output)

    except Exception as e:
        raise e
    finally:
        if hook is not None:
            hook.remove()

    cov_mat = estimator.compute()
    return cov_mat


class CovarianceEstimator:
    def __init__(self):
        self._cov_mat = None
        self._N = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        assert len(x.shape) in [2, 4]

        # check whehter we have Conv2D output
        if len(x.shape) == 4:
            x = utils.flatten_4d_tensor(x)

        N, _ = x.shape

        # todo: make it compatiblie with conv output
        curr_mat = (x.T @ x) / N

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
        assert self.cov_mat is not None

        return self.cov_mat.detach().cpu().numpy()
