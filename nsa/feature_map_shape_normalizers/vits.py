import torch

from .interface import FeatureMapShapeNormalizer


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
