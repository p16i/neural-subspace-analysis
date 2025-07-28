import typing
import torch
from torch import nn

from abc import ABC, abstractmethod


from nsa import utils, intercepts


class FeatureMapShapeNormalizer(ABC):
    @abstractmethod
    def to_cnn_shape(x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def to_original_shape(x: torch.Tensor) -> torch.Tensor:
        pass


class ViTFeatureMapShapeNormalizer(FeatureMapShapeNormalizer):
    @staticmethod
    def to_cnn_shape(x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 3

        x = x.permute(0, 2, 1).unsqueeze(3)
        return x

    @staticmethod
    def to_original_shape(x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 4
        b, d, ntokens, w = x.shape
        assert w == 1

        x = x.squeeze(3).permute(0, 2, 1)

        return x


class MLPFeatureMapShapeNormalizer(FeatureMapShapeNormalizer):
    @staticmethod
    def to_cnn_shape(x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 2

        x = x.unsqueeze(2).unsqueeze(3)
        return x

    @staticmethod
    def to_original_shape(x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 4

        x = x.squeeze(3).sequeeze(2)

        return x


@torch.no_grad()
def resolve_shape_normalizer(
    model: nn.Module,
    layer: str,
    dataloader: torch.utils.data.DataLoader,
) -> typing.Tuple[str, FeatureMapShapeNormalizer]:

    hook = None
    try:
        batch = next(iter(dataloader))
        x = utils.first_tensor_in_batch(batch)

        module = intercepts.get_module_for_layer(model=model, layer=layer)
        hook = module.register_forward_hook(intercepts.fh_intercept_output)

        model(x)

        output = intercepts.get_module_output(module)
        shape = output.shape

    finally:
        if hook is not None:
            hook.remove()

    if len(shape) == 4:
        return layer, None
    elif len(shape) == 2:
        return layer, MLPFeatureMapShapeNormalizer()
    elif len(shape) == 3:
        if "[cls]" in layer:
            layer = layer.replace("[cls]", "")
            raise NotImplementedError()
        else:
            return layer, ViTFeatureMapShapeNormalizer()
    else:
        raise ValueError(
            f"Unsupported shape {shape} for layer {layer}. "
            "Expected 1D, 3D, or 4D tensor shapes."
        )
