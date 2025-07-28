from abc import ABC, abstractmethod
from typing import List

from numpy import typing as npt
from torch.utils.data import DataLoader
from torch import nn

import pandas as pd


class EvaluatorWithLowRankProjection(ABC):
    """
    Abstract base class for evaluators.
    """

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
    def metric_keys(self) -> List[str]:
        pass
