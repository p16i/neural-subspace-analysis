import re
import typing

import numpy as np
import torch
from scipy.stats import ortho_group


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def first_tensor_in_batch(batch) -> torch.Tensor:
    x = batch[0] if isinstance(batch, typing.Sequence) else batch
    return x


def arange_with_grid(start: int, stop: int, step: int) -> list[int]:
    arr_ds = np.arange(start, stop, step=step)
    arr_ds_list = arr_ds.tolist()
    arr_ds_list = [*arr_ds_list, stop]

    return arr_ds_list


def eigh(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """_summary_

    Args:
        cov (torch.Tensor): _description_

    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]: _description_
    """

    d1, d2 = cov.shape

    if d1 != d2:
        raise ValueError(f"Covariance matrix must be square, shape= {cov.shape}")  # noqa: TRY003

    eigvals, eigvecs = torch.linalg.eigh(cov)

    if len(eigvals.shape) != 1:
        raise ValueError(f"Expected 1D eigenvalues, got shape {eigvals.shape}")  # noqa: TRY003

    eigvals = torch.flip(eigvals, dims=(0,))
    eigvecs = torch.flip(eigvecs, dims=(1,))

    return eigvals, eigvecs


# fixme : use torch tensor instead
# def _solve_eigh(cov: npt.NDArray) -> typing.Tuple[NDArray, NDArray]:
#     raise NotImplementedError("use torch tensor instead")
#     """Solve eigenvalues

#     Args:
#         x (_type_): _description_

#     Returns:
#         typing.Tuple[NDArray, NDArray]: _description_
#     """

#     d1, d2 = cov.shape
#     assert d1 == d2

#     eigvals, eigvecs = np.linalg.eigh(cov)

#     assert len(eigvals.shape) == 1

#     indices = np.argsort(-eigvals)
#     eigvals: NDArray = eigvals[indices]
#     eigvecs: NDArray = eigvecs[:, indices]

#     return eigvals, eigvecs


def reshape_tensor_to_cnn_like(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 4:
        return x
    elif len(x.shape) == 2:
        return x.unsqueeze(2).unsqueeze(3)

    else:
        raise ValueError(f"We don't support x.shape={x.shape}")  # noqa: TRY003


def flatten_4d_tensor(x: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        x (torch.Tensor): _description_

    Returns:
        torch.Tensor: 2d tensor whose len(x.shape) == 2
    """

    if len(x.shape) != 4:
        raise ValueError(f"Expected 4D tensor, got shape {x.shape}")  # noqa: TRY003

    flatenned_x = x.permute(1, 0, 2, 3).flatten(start_dim=1).T

    return flatenned_x


def parse_number_if_possible(text: str) -> typing.Union[None, int]:
    is_int = re.match(r"-?\d+", text) is not None

    if is_int:
        return int(text)
    else:
        return None


def parse_layers(txt: str, arr_default_layers: list[str]) -> list[str]:
    """
    Suppose arr_default_layers=["layer1","layer2","layer3","layer4"]

    Specifying txt="@2,layer2" yields ["layer3", "layer2"].
    Here, $i is resolved to the layer given in arr_default_layers

    Args:
        txt (str): _description_
        arr_default_layers (typing.List[str]): _description_

    Returns:
        typing.List[str]: _description_
    """

    slugs = txt.split(",")

    arr_layers = []

    for slug in slugs:
        if slug[0] == "@":
            layer_ix = int(slug[1:])
            arr_layers.append(arr_default_layers[layer_ix])
        else:
            arr_layers.append(slug)

    return arr_layers


def get_random_orthogonal_matrix(d: int, seed: int) -> torch.Tensor:
    """Generate a random orthogonal matrix of size d x d.

    Args:
        d (int): Dimension of the square matrix.
        seed (int): random seed

    Returns:
        torch.Tensor: A random orthogonal matrix of shape (d, d).
    """
    rng = np.random.default_rng(seed)
    Q = ortho_group.rvs(d, random_state=rng)
    return torch.from_numpy(Q).float()
