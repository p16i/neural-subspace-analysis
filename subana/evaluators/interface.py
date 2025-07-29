import typing
from abc import ABC, abstractmethod

import pandas as pd
import torch
from numpy import typing as npt
from torch import nn
from torch.utils.data import DataLoader


class EvaluatorWithLowRankProjection(ABC):
    """
    Abstract base class for evaluators.
    """

    feature_map_transform: typing.Optional[typing.Callable[[list[torch.Tensor]], torch.Tensor]]

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        layer: str,
        dataloader: DataLoader,
        U: npt.NDArray,
        arr_ks: npt.NDArray,
        device: str = "cpu",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Evaluate the model or data based on the implementation.
        """
        pass

    @property
    @abstractmethod
    def metric_keys(self) -> list[str]:
        pass
