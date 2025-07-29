import torch

from .interface import FeatureMapShapeNormalizer


class ViTFeatureMapShapeNormalizer(FeatureMapShapeNormalizer):
    @staticmethod
    def to_cnn_shape(x: torch.Tensor) -> torch.Tensor:
        """_summary_

        x.shape = (batch_size, num_patches, embed_dim)

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        assert len(x.shape) == 3  # noqa: S101

        x = x.permute(0, 2, 1).unsqueeze(3)
        return x

    @staticmethod
    def to_original_shape(x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 4  # noqa: S101
        b, d, ntokens, w = x.shape
        assert w == 1  # noqa: S101

        x = x.squeeze(3).permute(0, 2, 1)

        return x
