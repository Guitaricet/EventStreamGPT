# Ex lightning module functions
import dataclasses
from typing import Any

import torch

import torch
import torch.multiprocessing
import torch.utils.data
import torchmetrics
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelAveragePrecision,
)
from transformers import get_polynomial_decay_schedule_with_warmup

import wandb
from EventStream.config import MetricCategories, Metrics, MetricsConfig, OptimizationConfig, Split

from EventStream.data.types import DataModality
from EventStream.transformer.config import (
    Averaging,
    StructuredEventProcessingMode,
    StructuredTransformerConfig,
)
from EventStream.transformer.model_output import GenerativeSequenceModelOutput
from EventStream.transformer.utils import expand_indexed_regression, str_summary


class MetricsLogger:
    TRAIN_SKIP_METRICS = ("AUROC", "AUPRC", "per_class")
    CLASSIFICATION = {
        DataModality.SINGLE_LABEL_CLASSIFICATION,
        DataModality.MULTI_LABEL_CLASSIFICATION,
    }

    def __init__(
        self,
        config: StructuredTransformerConfig | dict[str, Any],
        metrics_config: MetricsConfig | dict[str, Any],
    ):
        """A class to handle metrics logging.

        Statefullness is mainly needed to store the metrics configuration.
        Distributed metric computation is not yet supported.
        """
        super().__init__()

        # If the configurations are dictionaries, convert them to class objects. They may be passed as
        # dictionaries when the lightning module is loaded from a checkpoint, so we need to support
        # this functionality.
        if type(config) is dict:
            config = StructuredTransformerConfig(**config)
        if type(metrics_config) is dict:
            metrics_config = MetricsConfig(**metrics_config)

        self.config = config
        self.metrics_config = metrics_config

        self.build_metrics()

    def build_metrics(self):
        """Build the various torchmetrics we'll use to track performance."""

        # For judging our ability to predict time-to-event, we'll use the following scores:
        #   1. Explained Variance
        #   2. Mean Squared Error
        #   3. Mean Squared Log Error
        self.tte_metrics = torch.nn.ModuleDict(
            {
                "MSE": torchmetrics.MeanSquaredError(),
                "MSLE": torchmetrics.MeanSquaredLogError(),
                "explained_variance": torchmetrics.ExplainedVariance(),
            }
        )

        self.metrics = torch.nn.ModuleDict()
        for task_type, measurements in self.config.measurements_per_generative_mode.items():
            for measurement in measurements:
                vocab_size = self.config.vocab_sizes_by_measurement[measurement]

                if measurement not in self.metrics:
                    self.metrics[measurement] = torch.nn.ModuleDict()
                if task_type not in self.metrics[measurement]:
                    self.metrics[measurement][task_type] = torch.nn.ModuleDict()

                match task_type:
                    case DataModality.SINGLE_LABEL_CLASSIFICATION:
                        cat = MetricCategories.CLASSIFICATION
                        metrics = {
                            Metrics.ACCURACY: (
                                MulticlassAccuracy,
                                [Averaging.MACRO, Averaging.WEIGHTED, Averaging.MICRO],
                            ),
                            Metrics.AUROC: (
                                MulticlassAUROC,
                                [Averaging.MACRO, Averaging.WEIGHTED],
                            ),
                            Metrics.AUPRC: (
                                MulticlassAveragePrecision,
                                [Averaging.MACRO, Averaging.WEIGHTED],
                            ),
                        }
                        metric_kwargs = {"num_classes": vocab_size, "ignore_index": 0}
                    case DataModality.MULTI_LABEL_CLASSIFICATION:
                        cat = MetricCategories.CLASSIFICATION
                        metrics = {
                            Metrics.ACCURACY: (
                                MultilabelAccuracy,
                                [Averaging.MACRO, Averaging.WEIGHTED, Averaging.MICRO],
                            ),
                            Metrics.AUROC: (
                                MultilabelAUROC,
                                [Averaging.MACRO, Averaging.WEIGHTED, Averaging.MICRO],
                            ),
                            Metrics.AUPRC: (
                                MultilabelAveragePrecision,
                                [Averaging.MACRO, Averaging.WEIGHTED, Averaging.MICRO],
                            ),
                        }
                        metric_kwargs = {"num_labels": vocab_size}
                    case DataModality.UNIVARIATE_REGRESSION:
                        cat = MetricCategories.REGRESSION
                        metrics = {
                            Metrics.MSE: (torchmetrics.MeanSquaredError, [None]),
                            Metrics.EXPLAINED_VARIANCE: (torchmetrics.ExplainedVariance, [None]),
                        }
                        metric_kwargs = {}
                    case DataModality.MULTIVARIATE_REGRESSION:
                        cat = MetricCategories.REGRESSION
                        metrics = {
                            Metrics.MSE: (torchmetrics.MeanSquaredError, [None]),
                            Metrics.EXPLAINED_VARIANCE: (
                                torchmetrics.ExplainedVariance,
                                [Averaging.MACRO, Averaging.WEIGHTED],
                            ),
                        }
                        metric_kwargs = {}
                    case _:
                        raise ValueError(f"Unrecognized modality {task_type}!")

                if not self.metrics_config.do_validate_args:
                    metric_kwargs["validate_args"] = False

                auc_kwargs = {**metric_kwargs, "thresholds": self.metrics_config.n_auc_thresholds}
                for metric, (metric_cls, averagings) in metrics.items():
                    if metric in (Metrics.AUROC, Metrics.AUPRC):
                        metric_cls_kwargs = {**auc_kwargs}
                    else:
                        metric_cls_kwargs = {**metric_kwargs}

                    for averaging in averagings:
                        if averaging is None:
                            metric_name = str(metric)
                            averaging_kwargs = {}
                        else:
                            metric_name = f"{averaging}_{metric}"
                            if metric == Metrics.EXPLAINED_VARIANCE:
                                if averaging == Averaging.MACRO:
                                    avg_str = "uniform_average"
                                elif averaging == Averaging.WEIGHTED:
                                    avg_str = "variance_weighted"
                                else:
                                    raise ValueError(f"{averaging} not supported for explained variance.")

                                averaging_kwargs = {"multioutput": avg_str}
                            else:
                                averaging_kwargs = {"average": averaging}

                        if self.metrics_config.do_log_any(cat, metric_name):
                            self.metrics[measurement][task_type][metric_name] = metric_cls(
                                **metric_cls_kwargs, **averaging_kwargs
                            )

    def _log_metric_dict(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        metrics: dict[str, torchmetrics.Metric],
        split: Split,
        measurement: str,
        cat: MetricCategories,
    ):
        """This helper function logs the set of named metrics for the predictions `preds` and labels `labels`.

        Args:
            `preds` (`torch.Tensor`): The predictions for this metric calculation.
            `labels` (`torch.Tensor`): The labels for this metric calculation.
            `metrics` (`Dict[str, torchmetrics.Metric]`): The metrics to log, by name.
            `skip_metrics` (`Sequence[str]`):
                A list of metrics to skip. Entries are not full metric names, but rather are partial names and
                any metric whose name contains an element of `skip_metrics` will be skipped.
                For example, if `skip_metrics = ['AUROC', 'AUPRC']`, then a metric with name `'macro_AUROC'`
                or `'micro_AUPRC'` would be skipped, whereas a metric named `'weighted_accuracy'` would not.
            `split` (`str`): TODO
            `measurement` (`str`): The measurement of this metric calculation. Affects the log name.
        """
        for metric_name, metric in metrics.items():
            # We'll want to skip a metric if any element of our skip_metrics list is a substring of the metric
            # name:
            if not self.metrics_config.do_log(split, cat, metric_name):
                continue

            try:
                if split != Split.TRAIN:
                    # This is slightly more efficient if we only care about epoch-level outputs.
                    # Source: https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
                    metric.update(preds, labels)
                else:
                    metric(preds, labels)

                self.log(
                    f"{split}/{measurement}_{metric_name}",
                    metric,
                    sync_dist=True,
                )
            except (ValueError, IndexError) as e:
                print(
                    f"Failed to compute {metric_name} for {measurement} "
                    f"with preds ({str_summary(preds)}) and labels ({str_summary(labels)}): {e}."
                )

    def log_tte_metrics(self, results: GenerativeSequenceModelOutput, split: Split):
        # The output of the model for time-to-event (and for regression targets as well) are pytorch
        # distribution objects, not scalars. So, for some evaluation metrics, we need to sample values from
        # those distributions to assess the metric.
        # TODO(mmd): We should likely be able to control how many samples are used, to minimize variance of
        # these results.
        tte_dist = results["preds"]["time_to_event"]
        tte_preds = tte_dist.sample()

        # After sampling, we also need to slice this down to just those intra-event-times that are actually
        # observed. This means we should drop the last sequence element (slice to `[:, :-1]` (as our observed
        # intra-event-times will only exist for the interior of our sequence), then further filter down to
        # just elements of the prediction for which the next sequence element was not masked
        # (mask via `results['event_mask'][:, 1:]`). We also need to filter the observed labels down to also
        # only be present for sequence elements where the next sequence element was truly observed.
        tte_preds = tte_preds[:, :-1][results["event_mask"][:, 1:]]
        tte_labels = results["labels"]["time_to_event"][results["event_mask"][:, 1:]]

        # Finally, we can log all relevant TTE metrics given these predictions and labels.
        self._log_metric_dict(
            preds=tte_preds,
            labels=tte_labels,
            metrics=self.tte_metrics,
            measurement="TTE",
            split=split,
            cat=MetricCategories.TTE,
        )

    def log_metrics(self, results: GenerativeSequenceModelOutput, step: int, split: Split):
        """Logs metric results for a given output result.

        Args:
            `results` (`transformerForGenerativeSequenceModelOutput`):
                The results to assess across the suite of metrics.
            `split` (`str`): The split that should be used when logging metric results.
        """

        # We always want to log the raw loss.
        self.log(f"{split}/loss", results["loss"], **log_kwargs)

        if self.metrics_config.do_log_only_loss(split):
            return

        # We start by logging the losses.
        self.log_dict(
            {f"{split}/{k}_cls_NLL": v for k, v in results["losses"]["classification"].items()},
            step=step,
        )
        self.log_dict(
            {f"{split}/{k}_reg_NLL": v for k, v in results["losses"]["regression"].items()},
            step=step,
        )
        self.log(f"{split}/TTE_reg_NLL", results["losses"]["time_to_event"], **log_kwargs)

        # Time-to-event
        if self.metrics_config.do_log(split, MetricCategories.TTE):
            self.log_tte_metrics(results, split)

        # Per data type
        for measurement, metrics_dict in self.metrics.items():
            mask = results["event_mask"]

            if not mask.any():
                continue

            for task_type, metrics in metrics_dict.items():
                if task_type in self.CLASSIFICATION and self.metrics_config.do_log(
                    split, MetricCategories.CLASSIFICATION
                ):
                    # For now, we ignore the is_observed distribution (the first element of the below tuple).
                    _, sample_dist = results["preds"]["classification"][measurement]
                    preds = sample_dist.logits
                    labels = results["labels"]["classification"][measurement]

                    # We need to filter these down to just those corresponding to observed events. Note that
                    # unlike TTE, the assumption here is that preds and labels correspond to predictions for
                    # and labels of the events at their indexed position; not for the subsequent event. So we
                    # don't need to shift `results['event_mask']` here to account for that.

                    preds = preds[mask]
                    labels = labels[mask].long()

                    self._log_metric_dict(
                        preds=preds,
                        labels=labels,
                        metrics=metrics,
                        measurement=measurement,
                        split=split,
                        cat=MetricCategories.CLASSIFICATION,
                    )

                elif task_type == DataModality.MULTIVARIATE_REGRESSION and self.metrics_config.do_log(
                    split, MetricCategories.REGRESSION
                ):
                    vocab_size = self.config.vocab_sizes_by_measurement[measurement]

                    # Here, like for TTE, we need to sample from the returned distribution before we can use
                    # it directly. Here we also need to limit to just those events that are actually observed.
                    # Like above, the assumption here is that preds and labels correspond to predictions for
                    # and labels of the events at their indexed position; not for the subsequent event. So we
                    # don't need to shift `results['event_mask']` here to account for that.
                    _, dist = results["preds"]["regression"][measurement]
                    preds = dist.sample()[mask]
                    labels = results["labels"]["regression"][measurement][mask]

                    # However, as our regression output is actually indexed only to the group keys that are
                    # actually measured (tracked in `results['preds']['regression_indices']`, we need to
                    # expand our predictions and labels to be in the full vocabulary space for the metrics to
                    # work naturally.
                    preds_indices = results["preds"]["regression_indices"][measurement][mask]
                    labels_indices = results["labels"]["regression_indices"][measurement][mask]

                    # We also need to reflect just those data elements for which values were observed:
                    data_el_mask = results["dynamic_values_mask"][mask]

                    preds = preds[data_el_mask]
                    labels = labels[data_el_mask]
                    preds_indices = preds_indices[data_el_mask]
                    labels_indices = labels_indices[data_el_mask]

                    preds_expanded = expand_indexed_regression(preds, preds_indices, vocab_size)
                    labels_expanded = expand_indexed_regression(labels, labels_indices, vocab_size)

                    self._log_metric_dict(
                        preds=preds_expanded,
                        labels=labels_expanded,
                        metrics=metrics,
                        measurement=measurement,
                        split=split,
                        cat=MetricCategories.REGRESSION,
                    )
                elif task_type == DataModality.UNIVARIATE_REGRESSION and self.metrics_config.do_log(
                    split, MetricCategories.REGRESSION
                ):
                    # Here, like for TTE, we need to sample from the returned distribution before we can use
                    # it directly. Here we also need to limit to just those events that are actually observed.
                    # Like above, the assumption here is that preds and labels correspond to predictions for
                    # and labels of the events at their indexed position; not for the subsequent event. So we
                    # don't need to shift `results['event_mask']` here to account for that.
                    # We ignore the is observed distribution here.
                    _, dist = results["preds"]["regression"][measurement]
                    preds = dist.sample()[mask]
                    labels = results["labels"]["regression"][measurement][mask]

                    self._log_metric_dict(
                        preds=preds,
                        labels=labels,
                        metrics=metrics,
                        measurement=measurement,
                        split=split,
                        cat=MetricCategories.REGRESSION,
                    )
