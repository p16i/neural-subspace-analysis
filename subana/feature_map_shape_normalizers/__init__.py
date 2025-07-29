import torch
from torch import nn

from subana import intercepts

from .interface import FeatureMapShapeNormalizer
from .mlps import MLPFeatureMapShapeNormalizer
from .vits import ViTFeatureMapShapeNormalizer


@torch.no_grad()
def resolve_shape_normalizer(
    model: nn.Module,
    layer: str,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> tuple[str, FeatureMapShapeNormalizer]:
    shape = intercepts.get_feature_map_shape(
        model=model,
        layer=layer,
        dataloader=dataloader,
        device=device,
    )

    if len(shape) == 4:
        return layer, None
    elif len(shape) == 2:
        return layer, MLPFeatureMapShapeNormalizer()
    elif len(shape) == 3:
        return layer, ViTFeatureMapShapeNormalizer()
    else:
        raise ValueError(f"Unsupported shape {shape} for layer {layer}. Expected 1D, 3D, or 4D tensor shapes.")  # noqa: TRY003
