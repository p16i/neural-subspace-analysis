from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torchmetrics import MeanMetric
from tqdm.autonotebook import tqdm

from subana import intercepts
from subana.feature_map_shape_normalizers import resolve_shape_normalizer
from subana.utils.metrics import ArrayMetric

from .interface import EvaluatorWithLowRankProjection


class ReconstructionErrorWithLowRankProjectionEvaluator(EvaluatorWithLowRankProjection):
    @property
    def metric_keys(self):
        return ["norm", "recon_err", "cossim"]

    def evaluate(self, model, layer, dataloader, U, arr_ks, device="cpu", verbose=True):
        d, _ = U.shape
        n = len(arr_ks)

        metric_norm = MeanMetric()
        arr_metric_norm = ArrayMetric(
            n=n,
            base_metric=MeanMetric(),
        )
        arr_metric_recon = ArrayMetric(
            n=n,
            base_metric=MeanMetric(),
        )
        arr_metric_cosine = ArrayMetric(
            n=n,
            base_metric=MeanMetric(),
        )

        layer, feature_map_shape_normalizer = resolve_shape_normalizer(
            model=model,
            layer=layer,
            dataloader=dataloader,
            device=device,
        )

        for batch in tqdm(
            dataloader,
            desc=f"[layer={layer}] evaluating reconstruction error",
            disable=not verbose,
        ):
            x = batch[0] if isinstance(batch, Sequence) else batch

            x = x.to(device)

            output = model(x).detach().cpu()  # Ensure logits are on CPU

            norm = torch.linalg.norm(output, ord=2, dim=1)
            metric_norm.update(norm)

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
                    recon_output = model(x).detach().cpu()

                    np.testing.assert_equal(len(output), len(recon_output))

                    err = torch.linalg.norm(output - recon_output, ord=2, dim=1)  # Compute reconstruction error
                    arr_metric_recon.update(kix, err.cpu())

                    # fixme add cosine
                    cosine_sim = torch.nn.functional.cosine_similarity(output, recon_output, dim=1)
                    arr_metric_cosine.update(kix, cosine_sim.cpu())
                finally:
                    if hook is not None:
                        hook.remove()

        # this just make sure we have the same data types
        arr_metric_norm = np.ones(n) * float(metric_norm.compute())

        arr_metric_recon = arr_metric_recon.compute()
        arr_metric_cosine = arr_metric_cosine.compute()

        data = dict(zip(["k", *self.metric_keys], [arr_ks, arr_metric_norm, arr_metric_recon, arr_metric_cosine]))

        df = pd.DataFrame(data=data)
        df["d"] = d
        return df
