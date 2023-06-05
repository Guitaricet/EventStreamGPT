"""Defines the interface for specifying functional time dependent measurements.

`EventStream.data.types.DataModality.FUNCTIONAL_TIME_DEPENDENT` measurements are specified by an analytical
function that depends only on the time of the event and per-subject static data. This module defines the
interface for specifying such functions, through the abstract base class `TimeDependentFunctor`.
The `AgeFunctor` and `TimeOfDayFunctor` classes are examples of such functions.
"""

from __future__ import annotations

import abc
from datetime import datetime
from typing import Any

import pandas as pd
import polars as pl
import torch

from .types import DataModality
from .vocabulary import Vocabulary


class TimeDependentFunctor(abc.ABC):
    """Abstract base class for specifying functional time dependent measurements.

    A functional time dependent measurement is specified by an analytical function that depends only on the
    time of the event and a subject's static data. It must be specified in functional form so that we can
    appropriately produce these measurements dynamically during generation. These functions must be computable
    in two ways:
      1. Via a `polars` expression that can be evaluated on a `polars.DataFrame` containing the static data
         and a `timestamp` column.
      2. Via a `torch` function that takes as input the prior timepoint's indices, values, and time, the time
         delta and time of the new event, and the vocabulary config and measurement metadata of a dataset, and
         returns the new indices and values of the output measurement.
    In addition, such functions must also be convertible to and from plain dictionaries, which must store the
    name of their class, for serializability. This is an abstract base class, and subclasses must overwrite
    the `pl_expr` and `update_from_prior_timepoint` functions to be valid.

    Attributes:
        OUTPUT_MODALITY: The `DataModality` of the output of the function.
    """

    OUTPUT_MODALITY: DataModality = DataModality.DROPPED

    def __init__(self, **fn_params):
        # Default to_dict/from_dict will only work if functions store all __init__ input params as class
        # member variables, and use those to compute the function values in __call__...
        for k, val in fn_params.items():
            setattr(self, k, val)

        self.link_static_cols = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "params": {k: v for k, v in vars(self).items() if k != "link_static_cols"},
        }

    @abc.abstractmethod
    def pl_expr(self) -> pl.Expression:
        """Defines the Polars Expression of the functor.

        This function must return a `polars` expression that computes the value of the functor when evaluated
        on a `polars.DataFrame` containing the static data and a `timestamp` column. This function must be
        overridden in subclasses.

        Raises:
            NotImplementedError: If this method is not overridden.
        """

        raise NotImplementedError("Must be implemented in subclass!")

    @abc.abstractmethod
    def update_from_prior_timepoint(
        self,
        prior_indices: torch.LongTensor,
        prior_values: torch.FloatTensor,
        new_delta: torch.FloatTensor,
        new_time: torch.FloatTensor,
        vocab: Vocabulary | None,
        measurement_metadata: pd.Series | None,
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        """Returns the pre-processed output for this measurement at a new timepoint.

        This method is used during generation to compute the output of the functor at a new timepoint (that
        has been stochastically generated by the model) given historical data and the timepoint of the new
        event. This method must be overridden in subclasses.

        Args:
            prior_indices: Prior timepoint indices.
            prior_values: Prior timepoint values.
            new_delta: Delta time until new event in minutes.
            new_time: Raw time in minutes of new event since 01/01/1970.
            vocab: Vocabulary config of a dataset.
            measurement_metadata: Metadata for the functional time dependent measurement as determined in
                pre-processing.

        Returns:
            Tuple of the new indices and values of the output measurement.

        Raises:
            NotImplementedError: If this method is not overridden.
        """
        raise NotImplementedError("Must be implemented in subclass!")

    @classmethod
    def from_dict(cls, in_dict: dict[str, Any]) -> TimeDependentFunctor:
        return cls(**in_dict["params"])

    def __eq__(self, other: TimeDependentFunctor) -> bool:
        return self.to_dict() == other.to_dict()


class AgeFunctor(TimeDependentFunctor):
    """Functor that returns the age of the subject when the event occurred.

    Note that as years are not a fixed unit of time, this measurement is returned in the average number of
    fixed-length years (where a fixed-length year is of length 365.25 days).

    Attributes:
        OUTPUT_MODALITY: `DataModality.UNIVARIATE_REGRESSION`
        dob_col: Column name containing the subject's date of birth.

    Example:
        >>> import polars as pl
        >>> from datetime import datetime
        >>> functor = AgeFunctor(dob_col="birth_date")
        >>> df = pl.DataFrame({
        ...     "birth_date": [datetime(1990, 1, 1), datetime(1995, 1, 1), datetime(2000, 1, 1)],
        ...     "timestamp": [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)],
        ... })
        >>> age_expr = functor.pl_expr().alias("age")
        >>> print(df.select(age_expr).get_column("age").to_list())
        [29.998631074606433, 26.001368925393567, 22.001368925393567]
    """

    OUTPUT_MODALITY: DataModality = DataModality.UNIVARIATE_REGRESSION

    def __init__(self, dob_col: str):
        self.dob_col = dob_col
        self.link_static_cols = [dob_col]

    def pl_expr(self) -> pl.Expression:
        return (pl.col("timestamp") - pl.col(self.dob_col)).dt.nanoseconds() / 1e9 / 60 / 60 / 24 / 365.25

    def update_from_prior_timepoint(
        self,
        prior_indices: torch.LongTensor,
        prior_values: torch.FloatTensor,
        new_delta: torch.FloatTensor,
        new_time: torch.FloatTensor,
        vocab: Vocabulary | None,
        measurement_metadata: pd.Series | None,
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        """Returns the pre-processed age for the subject at a new timepoint.

        This method is used during generation to compute the subject's age (appropriately pre-processed) at
        new timepoints (that have been stochastically generated by the model). The function infers the
        outlier detection and normalization parameters through the `measurement_metadata` argument.

        Args:
            prior_indices: Prior timepoint associated indices.
            prior_values: The subject's age (fully pre-processed) as of the last observed prior timepoint.
            new_delta: Delta time in minutes.
            new_time: Raw time in minutes since 01/01/1970.
            vocab: Vocabulary config of a dataset.
            measurement_metadata: Metadata for the age measurement as determined in pre-processing.

        Returns:
            The static index of the univariate age measurement and the new age of the subject at the new
            timepoint.
        """

        mean = measurement_metadata["normalizer"]["mean_"]
        std = measurement_metadata["normalizer"]["std_"]

        thresh_large = measurement_metadata["outlier_model"]["thresh_large_"]
        thresh_small = measurement_metadata["outlier_model"]["thresh_small_"]

        prior_age = (prior_values * std) + mean

        new_delta_yrs = new_delta / 60 / 24 / 365.25

        new_age = prior_age + new_delta_yrs

        new_age = torch.where(
            (new_age > thresh_large) | (new_age < thresh_small),
            float("nan") * torch.ones_like(new_age),
            new_age,
        )

        new_age = (new_age - mean) / std
        return prior_indices, new_age


class TimeOfDayFunctor(TimeDependentFunctor):
    """Functor that returns the time-of-day in 4 categories of when the event occurred.

    Attributes:
        OUTPUT_MODALITY: `DataModality.SINGLE_LABEL_CLASSIFICATION`

    Example:
        >>> import polars as pl
        >>> from datetime import datetime
        >>> functor = TimeOfDayFunctor()
        >>> df = pl.DataFrame({
        ...     "timestamp": [datetime(2020, 1, 1, 0, 0, 0), datetime(2020, 1, 1, 6, 0, 0),
        ...                   datetime(2020, 1, 1, 12, 0, 0), datetime(2020, 1, 1, 18, 0, 0),
        ...                   datetime(2020, 1, 1, 23, 59, 59)],
        ... })
        >>> time_of_day_expr = functor.pl_expr().alias("time_of_day")
        >>> print(df.select(time_of_day_expr).get_column("time_of_day").to_list())
        ['EARLY_AM', 'AM', 'PM', 'PM', 'LATE_PM']
    """

    OUTPUT_MODALITY: DataModality = DataModality.SINGLE_LABEL_CLASSIFICATION

    def pl_expr(self) -> pl.Expression:
        return (
            pl.when(pl.col("timestamp").dt.hour() < 6)
            .then("EARLY_AM")
            .when(pl.col("timestamp").dt.hour() < 12)
            .then("AM")
            .when(pl.col("timestamp").dt.hour() < 21)
            .then("PM")
            .otherwise("LATE_PM")
        )

    def update_from_prior_timepoint(
        self,
        prior_indices: torch.LongTensor,
        prior_values: torch.FloatTensor,
        new_delta: torch.FloatTensor,
        new_time: torch.FloatTensor,
        vocab: Vocabulary | None,
        measurement_metadata: pd.Series | None,
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        """Returns the pre-processed time of day for the subject at a new timepoint.

        This method is used during generation to compute the event's time of day, realized as an integer
        index, at new timepoints (that have been stochastically generated by the model). The function infers
        the vocabulary information from the `vocab` argument.

        Args:
            prior_indices: Prior timepoint associated indices in the global vocabulary.
            prior_values: An empty tensor (as this is a categorical measurement).
            new_delta: Delta time in minutes.
            new_time: Raw time in minutes since 01/01/1970.
            vocab: Vocabulary config of a dataset.
            measurement_metadata: `None`, as this is a categorical measurement.

        Returns:
            Tuple of the new indices of the subsequent time of day, and a tensor of `nan` values.
        """

        hrs_local_at_midnight_epoch = datetime(1970, 1, 1).timestamp() / 60 / 60

        # new time is in minutes since 01/01/1970 UTC
        new_hour_utc = new_time / 60
        new_hour_local = (new_hour_utc - hrs_local_at_midnight_epoch) % 24

        new_indices = torch.where(
            new_hour_local < 6,
            vocab.idxmap.get("EARLY_AM", 0),
            torch.where(
                new_hour_local < 12,
                vocab.idxmap.get("AM", 0),
                torch.where(new_hour_local < 21, vocab.idxmap.get("PM", 0), vocab.idxmap.get("LATE_PM", 0)),
            ),
        )

        return new_indices, float("nan") * prior_values
