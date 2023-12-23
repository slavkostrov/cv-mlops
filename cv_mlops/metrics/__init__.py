from typing import Iterable

import numpy as np

from .accuracy_score import accuracy
from .f1_score import f1

__all__ = ["f1", "accuracy", "calculate_metrics"]


def calculate_metrics(
    scores,
    labels,
    print_log: bool = False,
    metrics: Iterable[str] = ("f1", "accuracy"),
):
    """Compute all the metrics from tracked_metrics dict using scores and labels."""
    assert len(labels) == len(scores), print("Label and score lists are of different size")

    scores_array = np.array(scores).astype(np.float32)
    labels_array = np.array(labels)

    metric_results = {}
    for metric_name in metrics:
        metric_func = globals().get(metric_name)
        if metric_func is None:
            # TODO: fix with error log
            print("UNKNOWN METRIC")

        metric_value = metric_func(scores_array, labels_array)
        metric_results[metric_name] = metric_value

    if print_log:
        print(" | ".join(["{}: {:.4f}".format(k, v) for k, v in metric_results.items()]))

    return metric_results
