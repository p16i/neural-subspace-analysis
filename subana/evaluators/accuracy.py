import pandas as pd
from torch import nn
from torchmetrics import MeanMetric
from torchmetrics.classification import Accuracy
from tqdm.autonotebook import tqdm

from subana import intercepts
from subana.feature_map_shape_normalizers import (
    resolve_shape_normalizer,
)
from subana.utils.metrics import ArrayMetric

from .interface import EvaluatorWithLowRankProjection


class AccuracyWithLowRankProjectionEvaluator(EvaluatorWithLowRankProjection):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    @property
    def metric_keys(self):
        return ["acc", "xent"]

    def evaluate(self, model, layer, dataloader, U, arr_ks, device="cpu", verbose=True):
        d, _ = U.shape
        n = len(arr_ks)
        arr_metric_acc = ArrayMetric(
            n=n,
            base_metric=Accuracy(num_classes=self.num_classes, task="multiclass"),
        )
        arr_metric_xent = ArrayMetric(
            n=n,
            base_metric=MeanMetric(),
        )

        layer, feature_map_shape_normalizer = resolve_shape_normalizer(
            model=model,
            layer=layer,
            dataloader=dataloader,
            device=device,
        )

        for x, y in tqdm(dataloader, desc=f"[layer={layer}] evaluating accuracy", disable=not verbose):
            x = x.to(device)

            for kix, k in enumerate(arr_ks):
                Uk = U[:, :k]

                hook = None
                try:
                    module = intercepts.get_module_for_layer(model=model, layer=layer)
                    hook = module.register_forward_hook(
                        intercepts.construct_fh_with_projection(
                            Uk,
                            shape_normalizer=feature_map_shape_normalizer,
                            device=device,
                        )
                    )

                    logits = model(x).detach().cpu()

                    xent = nn.functional.cross_entropy(logits, y, reduction="none")
                    arr_metric_acc.update(kix, logits, y)
                    arr_metric_xent.update(kix, xent)
                finally:
                    if hook is not None:
                        hook.remove()

        arr_metric_acc = arr_metric_acc.compute()
        arr_metric_xent = arr_metric_xent.compute()

        data = dict(zip(["k", *self.metric_keys], [arr_ks, arr_metric_acc, arr_metric_xent]))

        df = pd.DataFrame(data=data)
        df["d"] = d
        return df
