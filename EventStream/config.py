import enum
import math
import dataclasses
from pathlib import Path
from typing import Any

import omegaconf

from .data.config import PytorchDatasetConfig
from .data.pytorch_dataset import PytorchDataset
from .transformer.config import (
    Averaging,
    StructuredTransformerConfig,
)
from .utils import JSONableMixin, StrEnum, hydra_dataclass

SKIP_CFG_PARAMS = {"seq_attention_layers", "dep_graph_attention_layers"}


@hydra_dataclass
class RetreiverConfig(JSONableMixin):
    faiss_index_path: str | None = None  # required if used
    dataset_path: str | None = None  # required if used
    muse_weights_path: str | None = None  # optional


@hydra_dataclass
class OptimizationConfig(JSONableMixin):
    """Configuration for optimization variables for training a model.

    Args:
        max_lr: The initial learning rate used by the optimizer. Given warmup is used, this will be the peak
            learning rate after the warmup period.
        end_lr: The final learning rate at the end of all learning rate decay.
        max_epochs: The maximum number of training epochs.
        batch_size: The batch size used during stochastic gradient descent.
        validation_batch_size: The batch size used during evaluation.
        lr_frac_warmup_steps: What fraction of the total training steps should be spent increasing the
            learning rate during the learning rate warmup period. Should not be set simultaneously with
            `lr_num_warmup_steps`. This is largely used in the `set_tot_dataset` function which initializes
            missing parameters given the dataset size, such as inferring the `max_num_training_steps` and
            setting `lr_num_warmup_steps` given this parameter and the inferred `max_num_training_steps`.
        lr_num_warmup_steps: How many training steps should be spent on learning rate warmup. If this is set
            then `lr_frac_warmup_steps` should be set to None, and `lr_frac_warmup_steps` will be properly
            inferred during `set_to_dataset`.
        max_training_steps: The maximum number of training steps the system will run for given `max_epochs`,
            `batch_size`, and the size of the used dataset (as inferred via `set_to_dataset`). Generally
            should not be set at initialization.
        lr_decay_power: The decay power in the learning rate polynomial decay with warmup. 1.0 corresponds to
            linear decay.
        weight_decay: The L2 weight regularization penalty that is applied during training.
        gradient_accumulation: The number of gradient accumulation steps to use. If None, gradient
            accumulation is not used.

    Raises:
        ValueError: If `end_lr`, `max_lr`, and `end_lr_frac_of_max_lr` are not consistent, or if `end_lr`
            and `end_lr_frac_of_max_lr` are both unset.
    """

    max_epochs: int = 1
    max_lr: float = 1e-3
    end_lr: float | None = None
    end_lr_frac_of_max_lr: float | None = 1e-3
    total_batch_size: int = 32
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    lr_frac_warmup_steps: float | None = 0.01
    lr_num_warmup_steps: int | None = None
    max_training_steps: int | None = None
    lr_decay_power: float = 1.0
    weight_decay: float = 0.01
    gradient_accumulation: int | None = None
    clip_grad_norm: float = 1.0
    eval_every: int = 1000
    save_every: int | None = 1000  # None for no saving
    keep_last_n_checkpoints: int | None = None  # keep all by default
    patience: int | None = None

    num_dataloader_workers: int = 0

    def __post_init__(self):
        if self.end_lr_frac_of_max_lr is not None:
            if self.end_lr_frac_of_max_lr <= 0.0 or self.end_lr_frac_of_max_lr >= 1.0:
                raise ValueError("`end_lr_frac_of_max_lr` must be between 0.0 and 1.0!")
            if self.end_lr is not None:
                prod = self.end_lr_frac_of_max_lr * self.max_lr
                if not math.isclose(self.end_lr, prod):
                    raise ValueError(
                        "If both set, `end_lr` must be equal to `end_lr_frac_of_max_lr * max_lr`! Got "
                        f"end_lr={self.end_lr}, end_lr_frac_of_max_lr * max_lr = {prod}!"
                    )
            self.end_lr = self.end_lr_frac_of_max_lr * self.max_lr
        else:
            if self.end_lr is None:
                raise ValueError("Must set either end_lr or end_lr_frac_of_max_lr!")
            self.end_lr_frac_of_max_lr = self.end_lr / self.max_lr

    def set_to_dataset(self, dataset: PytorchDataset):
        """Sets parameters in the config to appropriate values given dataset.

        Some parameters for optimization are dependent upon the total size of the dataset (e.g., converting
        between a fraction of training and a concrete number of steps). This function sets these parameters
        based on dataset's size.

        Args:
            dataset: The dataset to set the internal parameters too.

        Raises:
            ValueError: If the setting process does not yield consistent results.
        """

        steps_per_epoch = int(math.ceil(len(dataset) / self.total_batch_size))

        if self.max_training_steps is None:
            self.max_training_steps = steps_per_epoch * self.max_epochs

        if self.lr_num_warmup_steps is None:
            assert self.lr_frac_warmup_steps is not None
            self.lr_num_warmup_steps = int(round(self.lr_frac_warmup_steps * self.max_training_steps))
        elif self.lr_frac_warmup_steps is None:
            self.lr_frac_warmup_steps = self.lr_num_warmup_steps / self.max_training_steps

        if not (
            math.floor(self.lr_frac_warmup_steps * self.max_training_steps) <= self.lr_num_warmup_steps
        ) and (math.ceil(self.lr_frac_warmup_steps * self.max_training_steps) >= self.lr_num_warmup_steps):
            raise ValueError(
                "`self.lr_frac_warmup_steps`, `self.max_training_steps`, and `self.lr_num_warmup_steps` "
                "should be consistent, but they aren't! Got\n"
                f"\tself.max_training_steps = {self.max_training_steps}\n"
                f"\tself.lr_frac_warmup_steps = {self.lr_frac_warmup_steps}\n"
                f"\tself.lr_num_warmup_steps = {self.lr_num_warmup_steps}"
            )


class Split(StrEnum):
    """What data split is being used."""

    TRAIN = enum.auto()
    """The train split."""

    TUNING = enum.auto()
    """The hyperparameter tuning split.

    Also often called "dev", "validation", or "val".
    """

    HELD_OUT = enum.auto()
    """The held out test set split.

    Also often called "test".
    """


class MetricCategories(StrEnum):
    """Describes different categories of metrics.

    Used for configuring what metrics to track.
    """

    TTE = "TTE"
    """Track metrics related to time-to-event prediction."""

    CLASSIFICATION = enum.auto()
    """Track metrics for generative prediction of classification metrics."""

    REGRESSION = enum.auto()
    """Track metrics for generative prediction of regression metrics."""


class Metrics(StrEnum):
    """Describes the different supported metric functions."""

    AUROC = "AUROC"
    """The area under the receiver operating characteristic.

    Also commonly called "AUC".
    """

    AUPRC = "AUPRC"
    """The area under the precision recall curve.

    Also commonly refferred to as "Average Precision".
    """

    ACCURACY = enum.auto()
    """Raw accuracy."""

    EXPLAINED_VARIANCE = enum.auto()
    """The extent to which the predicted regression label explains the variance in the true label."""

    MSE = "MSE"
    """The mean squared error between predicted and true regression labels."""

    MSLE = "MSLE"
    """The mean squared log error between predicted and true regression labels."""


@hydra_dataclass
class MetricsConfig(JSONableMixin):
    """An overall configuration for what metrics should be tracked.

    Args:
        n_auc_thresholds: The number of thresholds to be used when computing AUROCs, for memory efficiency.
        do_skip_all_metrics: If `True`, all metrics will be skipped by the model. This can save significant
            time.
        do_validate_args: If `True`, `torchmetrics` metrics objects will validate their arguments during
            computation. This costs time.
        include_metrics: A dictionary detailing what metrics should be tracked over what splits, for what
            measurements, in what ways. If `do_skip_all_metrics`, this will be silently overwritten with {}.
            The format for this dictionary is as follows. The outermost level of keys is splits. Within each
            split, there is another dictionary, whose keys are metric categories that should be tracked in
            some form on that split. Each metric category maps to either the boolean `True`, in which case
            that metric category should be tracked across all relevant metrics, or to a dictionary mapping
            metric functions to either the boolean `True`, indicating they should be tracked over all relevant
            weightings, or to a list of weightings which should be tracked.
    """

    n_auc_thresholds: int | None = 50
    do_skip_all_metrics: bool = False
    do_validate_args: bool = False

    include_metrics: dict[
        # Split, Dict[MetricCategories, Union[bool, Dict[Metrics, Union[bool, List[Averaging]]]]]
        str,
        Any,
    ] = dataclasses.field(
        default_factory=lambda: (
            {
                Split.TUNING: {
                    MetricCategories.TTE: {Metrics.MSE: True, Metrics.MSLE: True},
                    MetricCategories.CLASSIFICATION: {
                        Metrics.AUROC: [Averaging.WEIGHTED],
                        Metrics.ACCURACY: True,
                    },
                    MetricCategories.REGRESSION: {Metrics.MSE: True},
                },
                Split.HELD_OUT: {
                    MetricCategories.TTE: {Metrics.MSE: True, Metrics.MSLE: True},
                    MetricCategories.CLASSIFICATION: {
                        Metrics.AUROC: [Averaging.WEIGHTED],
                        Metrics.ACCURACY: True,
                    },
                    MetricCategories.REGRESSION: {Metrics.MSE: True},
                },
            }
        )
    )

    def __post_init__(self):
        if self.do_skip_all_metrics:
            self.include_metrics = {}

    def do_log_only_loss(self, split: Split) -> bool:
        """Returns True if only loss should be logged for this split."""
        if (
            self.do_skip_all_metrics
            or split not in self.include_metrics
            or not self.include_metrics[split]
        ):
            return True
        else:
            return False

    def do_log(self, split: Split, cat: MetricCategories, metric_name: str | None = None) -> bool:
        """Returns True if `metric_name` should be tracked for `split` and `cat`."""
        if self.do_log_only_loss(split):
            return False

        inc_dict = self.include_metrics[split].get(cat, False)
        if not inc_dict:
            return False
        elif metric_name is None or inc_dict is True:
            return True

        has_averaging = "_" in metric_name.replace("explained_variance", "")
        if not has_averaging:
            return metric_name in inc_dict

        parts = metric_name.split("_")
        averaging = parts[0]
        metric = "_".join(parts[1:])

        permissible_averagings = inc_dict.get(metric, [])
        if (permissible_averagings is True) or (averaging in permissible_averagings):
            return True
        else:
            return False

    def do_log_any(self, cat: MetricCategories, metric_name: str | None = None) -> bool:
        """Returns True if `metric_name` should be tracked for `cat` and any split."""
        for split in Split.values():
            if self.do_log(split, cat, metric_name):
                return True
        return False


@hydra_dataclass
class PretrainConfig:
    do_overwrite: bool = False
    seed: int = 1

    model_config: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "_target_": "EventStream.transformer.config.StructuredTransformerConfig",
            **{
                k: v
                for k, v in StructuredTransformerConfig(measurements_per_dep_graph_level=[]).to_dict().items()
                if k not in SKIP_CFG_PARAMS
            },
        }
    )
    optimization_config: OptimizationConfig = OptimizationConfig()
    data_config: PytorchDatasetConfig = PytorchDatasetConfig()
    pretraining_metrics_config: MetricsConfig = MetricsConfig(do_skip_all_metrics=True)
    final_validation_metrics_config: MetricsConfig = MetricsConfig(do_skip_all_metrics=False)
    retreiver_config: RetreiverConfig = RetreiverConfig()

    trainer_config: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "accelerator": "auto",
            "devices": "auto",
            "detect_anomaly": False,
            "default_root_dir": "${save_dir}/model_checkpoints",
            "log_every_n_steps": 10,
        }
    )

    experiment_dir: str = omegaconf.MISSING
    save_dir: str = "${experiment_dir}/pretrain/${now:%Y-%m-%d_%H-%M-%S}"

    wandb_logger_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "project": "EventStream",
        }
    )

    num_dataloader_workers: int = 1

    do_final_validation_on_metrics: bool = True

    # compile: bool = True

    def __post_init__(self):
        if type(self.save_dir) is str and self.save_dir != omegaconf.MISSING:
            self.save_dir = Path(self.save_dir)
        if "max_epochs" in self.trainer_config:
            raise ValueError("Max epochs is set in the optimization_config, not the trainer config!")
        if "callbacks" in self.trainer_config:
            raise ValueError("Callbacks are built internally, not set via trainer_config!")
