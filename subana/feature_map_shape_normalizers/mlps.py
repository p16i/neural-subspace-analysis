import torch

from .interface import FeatureMapShapeNormalizer


class MLPFeatureMapShapeNormalizer(FeatureMapShapeNormalizer):
    @staticmethod
    def to_cnn_shape(x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 2  # noqa: S101

        x = x.unsqueeze(2).unsqueeze(3)
        return x

    @staticmethod
    def to_original_shape(x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 4  # noqa: S101

        x = x.squeeze(3).squeeze(2)

        return x
