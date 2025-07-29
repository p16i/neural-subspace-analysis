from copy import deepcopy

import numpy as np
import numpy.typing as npt
from torchmetrics import Metric


class ArrayMetric(Metric):
    def __init__(self, n: int, base_metric: Metric):
        self.n = n
        self.basket = []
        for _ in range(n):
            self.basket.append(deepcopy(base_metric))

    def update(self, ix: int, *args):
        self.basket[ix].update(*args)

    def compute(self) -> npt.NDArray:
        arr = []
        for i in range(self.n):
            value = self.basket[i].compute()
            arr.append(value)

        return np.array(arr, dtype=float)
