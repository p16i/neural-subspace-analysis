from abc import ABC, abstractmethod

import torch


class FeatureMapShapeNormalizer(ABC):
    @abstractmethod
    def to_cnn_shape(x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def to_original_shape(x: torch.Tensor) -> torch.Tensor:
        pass
