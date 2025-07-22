import typing

import torch
from torch import nn
from torch.nn import functional as F


from nsa import utils


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
            assert isinstance(parent_module, nn.Sequential)
            assert isinstance(parsed_attr_name, int)

            parent_module = parent_module[parsed_attr_name]
        else:
            parent_module = getattr(parent_module, attr_name)

    module = parent_module

    return module


def construct_fh_with_projection(U: torch.Tensor, device="cpu") -> typing.Callable:

    d, K = U.shape

    assert d >= K

    UUT = (U @ U.T).unsqueeze(2).unsqueeze(3).to(device)

    def fh(mod, inp, out):
        orig_shape = out.shape
        out = utils.reshape_tensor_to_cnn_like(out)

        out = F.conv2d(out, UUT)

        return out.reshape(orig_shape)

    return fh
