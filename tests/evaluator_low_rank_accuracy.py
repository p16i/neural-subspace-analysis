import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from nsa.evaluators.accuracy import AccuracyWithLowRankProjectionEvaluator


class DummyModel(torch.nn.Module):
    def __init__(self, W=None, b=None):
        super().__init__()
        self.lin1 = torch.nn.Conv2d(4, 10, kernel_size=2)
        self.act1 = torch.nn.ReLU()
        self.avg = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.lin2 = torch.nn.Linear(10, 3)

    def forward(self, x):
        out = self.act1(self.lin1(x))
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        return self.lin2(out)


def test_accuracy_with_low_rank_projection_evaluator():
    # Create dummy data: 10 samples, 4 features, 3 classes
    X = torch.randn(10, 4, 28, 28)  # 4 channels, 8x8 spatial dimensions
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=5)

    # Dummy projection matrix: identity (no projection)
    U = torch.eye(10)
    arr_ks = [2, 4]  # test for k=2 and k=4

    # Model: simple linear classifier
    model = DummyModel()

    evaluator = AccuracyWithLowRankProjectionEvaluator(num_classes=3)

    df = evaluator.evaluate(
        model,
        layer="lin1",
        dataloader=dataloader,
        U=U,
        arr_ks=np.array(arr_ks),
        device="cpu",
        verbose=False,
    )

    assert df.shape == (2, 3)  # 2 ks, 3 metrics

    assert df.columns.tolist() == ["k", *evaluator.metric_keys]

    assert np.all(df["acc"] >= 0)
    assert np.all(df["xent"] >= 0)

    np.testing.assert_equal(df["k"].values, arr_ks)  # Check if k values are as expected
