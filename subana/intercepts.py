import typing

import torch
from torch import nn
from torch.nn import functional as F

from subana import utils
from subana.feature_map_shape_normalizers.interface import FeatureMapShapeNormalizer

ATTRIBUTE_OUTPUT_KEY = "__output"


def fh_intercept_output(mod, inp, outp: torch.Tensor):
    setattr(mod, ATTRIBUTE_OUTPUT_KEY, outp)


def fh_intercept_output_and_retain_grad(mod, inp, outp: torch.Tensor):
    outp.retain_grad()

    setattr(mod, ATTRIBUTE_OUTPUT_KEY, outp)


def get_module_output(module) -> torch.Tensor:
    return getattr(module, ATTRIBUTE_OUTPUT_KEY)


def get_module_for_layer(model: nn.Module, layer: str) -> nn.Module:
    arr_level_layers = layer.split(".")

    parent_module = model

    for attr_name in arr_level_layers:
        parsed_attr_name = utils.parse_number_if_possible(attr_name)

        if parsed_attr_name is not None:
            if not isinstance(parent_module, nn.Sequential):
                raise ValueError(f"Expected nn.Sequential, got {type(parent_module)}")  # noqa: TRY003

            if not isinstance(parsed_attr_name, int):
                raise ValueError(f"Expected integer index, got {parsed_attr_name} of type {type(parsed_attr_name)}")  # noqa: TRY003

            parent_module = parent_module[parsed_attr_name]
        else:
            parent_module = getattr(parent_module, attr_name)

    module = parent_module

    return module


def construct_fh_with_projection(
    U: torch.Tensor,
    shape_normalizer: typing.Optional[FeatureMapShapeNormalizer] = None,
    device="cpu",
) -> typing.Callable:
    d, K = U.shape

    if not (d >= K):
        raise ValueError(f"Expected d >= K, got d={d}, K={K}")  # noqa: TRY003

    UUT = (U @ U.T).unsqueeze(2).unsqueeze(3).to(device)

    def fh(mod, inp, out):
        orig_shape = out.shape

        # canonicalize the featuremap shape
        if shape_normalizer is not None:
            out = shape_normalizer.to_cnn_shape(out)

        out = F.conv2d(out, UUT)

        # decanonicalize the featuremap shape
        if shape_normalizer is not None:
            out = shape_normalizer.to_original_shape(out)
            if out.shape != orig_shape:
                raise ValueError(f"Expected shape {orig_shape}, got {out.shape}")  # noqa: TRY003

        return out.reshape(orig_shape)

    return fh


def get_feature_map_shape(
    model: nn.Module,
    layer: str,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> torch.Size:
    hook = None
    try:
        batch = next(iter(dataloader))
        x = utils.first_tensor_in_batch(batch)

        module = get_module_for_layer(model=model, layer=layer)
        hook = module.register_forward_hook(fh_intercept_output)

        model(x.to(device))

        output = get_module_output(module)
        shape = output.shape

    finally:
        if hook is not None:
            hook.remove()

    return shape
