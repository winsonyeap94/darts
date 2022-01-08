"""
Functions for calculating forecast accuracies based on VDA 9000.

The functions are designed to be able to work with the darts framework.

For reference of how metrics are designed in the darts framework, refer:
https://github.com/unit8co/darts/blob/8fa7dbfa48577587ba247244b381c48d01153a70/darts/metrics/metrics.py
"""

import numpy as np
import pandas as pd
from functools import wraps
from darts import TimeSeries
from inspect import signature
from datetime import timedelta
from darts.logging import raise_if_not, get_logger, raise_log
from typing import AsyncIterable, Optional, Callable, Sequence, Union, Tuple
from darts.utils import _parallel_apply, _build_tqdm_iterator

logger = get_logger(__name__)


# =============================================================================
# DARTS DECORATORS
# =============================================================================
# Note: for new metrics added to this module to be able to leverage the two decorators, it is required both having
# the `actual_series` and `pred_series` parameters, and not having other ``Sequence`` as args (since these decorators
# don't “unpack“ parameters different from `actual_series` and `pred_series`). In those cases, the new metric must take
# care of dealing with Sequence[TimeSeries] and multivariate TimeSeries on its own (See mase() implementation).


def multi_ts_support(func):
    """
    This decorator further adapts the metrics that took as input two univariate/multivariate ``TimeSeries`` instances,
    adding support for equally-sized sequences of ``TimeSeries`` instances. The decorator computes the pairwise metric for
    ``TimeSeries`` with the same indices, and returns a float value that is computed as a function of all the
    pairwise metrics using a `inter_reduction` subroutine passed as argument to the metric function.
    If a 'Sequence[TimeSeries]' is passed as input, this decorator provides also parallelisation of the metric
    evaluation regarding different ``TimeSeries`` (if the `n_jobs` parameter is not set 1).
    """

    @wraps(func)
    def wrapper_multi_ts_support(*args, **kwargs):
        actual_series = kwargs['actual_series'] if 'actual_series' in kwargs else args[0]
        pred_series = kwargs['pred_series'] if 'pred_series' in kwargs else args[0] if 'actual_series' in kwargs \
            else args[1]

        n_jobs = kwargs.pop('n_jobs', signature(func).parameters['n_jobs'].default)
        verbose = kwargs.pop('verbose', signature(func).parameters['verbose'].default)

        raise_if_not(isinstance(n_jobs, int), "n_jobs must be an integer")
        raise_if_not(isinstance(verbose, bool), "verbose must be a bool")

        actual_series = [actual_series] if not isinstance(actual_series, Sequence) else actual_series
        pred_series = [pred_series] if not isinstance(pred_series, Sequence) else pred_series

        raise_if_not(len(actual_series) == len(pred_series),
                     "The two TimeSeries sequences must have the same length.", logger)

        num_series_in_args = int('actual_series' not in kwargs) + int('pred_series' not in kwargs)
        kwargs.pop('actual_series', 0)
        kwargs.pop('pred_series', 0)

        iterator = _build_tqdm_iterator(iterable=zip(actual_series, pred_series),
                                        verbose=verbose,
                                        total=len(actual_series))

        value_list = _parallel_apply(iterator=iterator,
                                     fn=func,
                                     n_jobs=n_jobs,
                                     fn_args=args[num_series_in_args:],
                                     fn_kwargs=kwargs)

        # in case the reduction is not reducing the metrics sequence to a single value, e.g., if returning the
        # np.ndarray of values with the identity function, we must handle the single TS case, where we should
        # return a single value instead of a np.array of len 1

        if len(value_list) == 1:
            value_list = value_list[0]

        if 'inter_reduction' in kwargs:
            return kwargs['inter_reduction'](value_list)
        else:
            return signature(func).parameters['inter_reduction'].default(value_list)

    return wrapper_multi_ts_support


def multivariate_support(func):
    """
    This decorator transforms a metric function that takes as input two univariate TimeSeries instances
    into a function that takes two equally-sized multivariate TimeSeries instances, computes the pairwise univariate
    metrics for components with the same indices, and returns a float value that is computed as a function of all the
    univariate metrics using a `reduction` subroutine passed as argument to the metric function.
    """
    @wraps(func)
    def wrapper_multivariate_support(*args, **kwargs):
        # we can avoid checks about args and kwargs since the input is adjusted by the previous decorator
        actual_series = args[0]
        pred_series = args[1]

        raise_if_not(actual_series.width == pred_series.width,
                     "The two TimeSeries instances must have the same width.", logger)

        value_list = []
        for i in range(actual_series.width):
            value_list.append(func(actual_series.univariate_component(i), pred_series.univariate_component(i),
                              *args[2:], **kwargs))  # [2:] since we already know the first two arguments are the series
        if 'reduction' in kwargs:
            return kwargs['reduction'](value_list)
        else:
            return signature(func).parameters['reduction'].default(value_list)
    return wrapper_multivariate_support


# =============================================================================
# DARTS AUXILIARY FUNCTIONS
# =============================================================================
def _get_values(series: TimeSeries,
                stochastic_quantile: Optional[float] = 0.5) -> np.ndarray:
    """
    Returns the numpy values of a time series.
    For stochastic series, return either all sample values with (stochastic_quantile=None) or the quantile sample value
    with (stochastic_quantile {>=0,<=1})
    """
    if series.is_deterministic:
        series_values = series.univariate_values()
    else:  # stochastic
        if stochastic_quantile is None:
            series_values = series.all_values(copy=False)
        else:
            series_values = series.quantile_timeseries(quantile=stochastic_quantile).univariate_values()
    return series_values


def _get_values_or_raise(series_a: TimeSeries,
                         series_b: TimeSeries,
                         intersect: bool,
                         stochastic_quantile: Optional[float] = 0.5,
                         remove_nan_union: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the processed numpy values of two time series. Processing can be customized with arguments
    `intersect, stochastic_quantile, remove_nan_union`.
    Raises a ValueError if the two time series (or their intersection) do not have the same time index.
    Parameters
    ----------
    series_a
        A univariate deterministic ``TimeSeries`` instance (the actual series).
    series_b
        A univariate (deterministic or stochastic) ``TimeSeries`` instance (the predicted series).
    intersect
        A boolean for whether or not to only consider the time intersection between `series_a` and `series_b`
    stochastic_quantile
        Optionally, for stochastic predicted series, return either all sample values with (`stochastic_quantile=None`)
        or any deterministic quantile sample values by setting `stochastic_quantile=quantile` {>=0,<=1}.
    remove_nan_union
        By setting `remove_non_union` to True, remove all indices from `series_a` and `series_b` which have a NaN value
        in either of the two input series.
    """

    raise_if_not(series_a.width == series_b.width, " The two time series must have the same number of components",
                 logger)

    raise_if_not(isinstance(intersect, bool), "The intersect parameter must be a bool")

    series_a_common = series_a.slice_intersect(series_b) if intersect else series_a
    series_b_common = series_b.slice_intersect(series_a) if intersect else series_b

    raise_if_not(series_a_common.has_same_time_as(series_b_common), 'The two time series (or their intersection) '
                                                                    'must have the same time index.'
                                                                    '\nFirst series: {}\nSecond series: {}'.format(
                                                                    series_a.time_index, series_b.time_index),
                 logger)

    series_a_det = _get_values(series_a_common, stochastic_quantile=stochastic_quantile)
    series_b_det = _get_values(series_b_common, stochastic_quantile=stochastic_quantile)

    if not remove_nan_union:
        return series_a_det, series_b_det

    b_is_deterministic = bool(len(series_b_det.shape) == 1)
    if b_is_deterministic:
        isnan_mask = np.logical_or(np.isnan(series_a_det), np.isnan(series_b_det))
    else:
        isnan_mask = np.logical_or(np.isnan(series_a_det), np.isnan(series_b_det).any(axis=2).flatten())
    return np.delete(series_a_det, isnan_mask), np.delete(series_b_det, isnan_mask, axis=0)


# =============================================================================
# VDA 9000 METRICS
# =============================================================================
def fai_dfu(actual_series: TimeSeries,
            pred_series: Sequence[TimeSeries],
            weighting_factors: Sequence[float],
            ignore_incomplete_lags: bool = True) -> Union[float, np.ndarray]:
    """
    Forecast Accuracy Index (FAI) metric calculation.

    Note:
    1. This assumes that the predictions are of a complete series (e.g., if Prediction A has 12 predictions ahead, 
    Prediction B should have the same too).

    Args:
        actual_series (TimeSeries): 
            The (sequence of) actual series.
        pred_series (Sequence[TimeSeries]): 
            The (sequence of) predicted series.
        weighting_factors (Sequence[float]):
            List of weights in the order of lag_1 to lag_n (based on length of list passed).
        ignore_incomplete_lags (bool):
            Whether to exclude timestamps which are do not have complete lags for the metric. Defaults to True.

    Returns:
        Union[float, np.ndarray]: The Forecast Accuracy Index (FAI) at a DFU (non-rolled up) level.
    """

    # TODO: What if there is a gap in the data?

    # Input argument check
    assert type(actual_series) is not list, "The actual series must be a single ``TimeSeries`` instance"
    assert type(pred_series) is list, "The predicted series must be a list of ``TimeSeries`` instances"
    assert type(weighting_factors) is list, "The weighting factors must be a list of floats"

    # Arranging pred_series in the order of start-of-forecast timestamps
    pred_start_timestamps = [x.start_time() for x in pred_series]
    pred_series = [x for _, x in sorted(zip(pred_start_timestamps, pred_series))]

    # Identifying relevant timestamps for metric calculation
    relevant_timestamps = None
    for pred_ts in pred_series:
        if relevant_timestamps is None:
            relevant_timestamps = pred_ts.time_index
        else:
            relevant_timestamps = relevant_timestamps.union(pred_ts.time_index)

    # # Filter to ignore incomplete lags
    # if ignore_incomplete_lags:
    #     # We ignore timestamps where (Number of predictions) != (Number of weighting factors)
    #     relevant_counts = np.unique(relevant_timestamps, return_counts=True)
    #     relevant_counts = zip(relevant_counts[0], relevant_counts[1])
    #     relevant_timestamps = list(filter(lambda x: x[1] >= len(weighting_factors), relevant_counts))
    #     relevant_timestamps = [x[0] for x in relevant_timestamps]
    # else:
    #     relevant_timestamps = list(set(relevant_timestamps))

    # Only selecting timestamps which match with actual_series (for accuracy measurement purposes)
    relevant_timestamps = relevant_timestamps.intersection(actual_series.time_index)

    # Consolidating actual and predicted TimeSeries into a single dataframe of shape 
    # (relevant_timestamps, 'date' + 'actual' + number of predicted series)
    fai_df = pd.DataFrame({
        'date': actual_series[relevant_timestamps].time_index.values,
        'actual': actual_series[relevant_timestamps].values().flatten(),
    })
    for idx, pred_ts in enumerate(pred_series):
        pred_df = pd.DataFrame({
            'date': pred_ts.time_index,
            f'forecast_{idx + 1}': pred_ts.values().flatten(),
        })
        fai_df = fai_df.merge(pred_df, how='left', on='date')

    # Rearranging fai_df into the shape of (relevant_timestamps, n_lags + 'date'+ 'actual') 
    def reshape_lags(group_df, n_lags=None):
        """Reshape groups of data (each date row) so that columns are now lags."""
        # Getting only forecast columns
        group_forecast_df = group_df.drop(columns=['date', 'actual'])
        # If n_lags is not provided, estimate from number of forecast columns.
        # Purpose of this is to speed-up runtime if it is provided upfront.
        if n_lags is None:
            n_lags = group_forecast_df.shape[1]
        # Since forecasts are arranged in the order of earliest to latest, we need to reverse the order
        forecast_values = group_forecast_df.values[0][::-1]
        forecast_values = forecast_values[~np.isnan(forecast_values)]
        # Pad numpy array with NaNs to make it of length n_lags
        padded_forecast_values = np.empty(n_lags)
        padded_forecast_values[:] = np.nan
        padded_forecast_values[:len(forecast_values)] = forecast_values
        # Arranging them back into a dataframe to be returned
        forecast_df = pd.DataFrame(padded_forecast_values).transpose().reset_index(drop=True)
        forecast_df.columns = [f"lag_{x}" for x in range(1, n_lags + 1)]
        return forecast_df
        
    fai_df = fai_df.groupby(['date', 'actual']).apply(lambda x: reshape_lags(x)).reset_index(drop=False)
    fai_df = fai_df.drop(columns=[x for x in fai_df.columns if x.startswith('level')])

    # If ignore_incomplete_lags is True, then we need to remove rows where (Number of forecasts) != (Number of weights)
    if ignore_incomplete_lags:
        fai_df = fai_df.dropna(axis=0, thresh=len(weighting_factors) + 2)  # 2 for 'date' and 'actual' columns

    # Dropping columns which are not used based on number of weights
    lag_columns = [f'lag_{x}' for x in range(1, len(weighting_factors) + 1)]
    fai_df = fai_df[['date', 'actual'] + lag_columns]

    # Calculating delta and abs_delta
    delta_columns = [f'delta_{x}' for x in range(1, len(weighting_factors) + 1)]
    abs_delta_columns = [f'abs_delta_{x}' for x in range(1, len(weighting_factors) + 1)]
    weighted_factor_columns = [f'weighted_factor_{x}' for x in range(1, len(weighting_factors) + 1)]
    weighted_fai_df = fai_df.copy()
    weighted_fai_df[delta_columns] = weighted_fai_df[lag_columns].apply(lambda x: x - weighted_fai_df['actual'], axis=0)
    weighted_fai_df[abs_delta_columns] = weighted_fai_df[delta_columns].apply(lambda x: np.abs(x), axis=0)

    # Calculating FAI
    weighted_fai_df[weighted_factor_columns] = 1 - weighted_fai_df[abs_delta_columns]\
        .apply(lambda x: x / weighted_fai_df['actual'], axis=0)
    weighted_fai_df[weighted_factor_columns].apply(lambda x: x.apply(lambda y: max(0, y)), axis=0)
    weighted_fai_df[weighted_factor_columns] = weighted_fai_df[weighted_factor_columns]\
        .apply(lambda x: np.clip(x, 0, None), axis=0)

    weighted_fai_df[weighted_factor_columns] = weighted_fai_df[weighted_factor_columns] * weighting_factors
    weighted_fai_df['fai'] = weighted_fai_df[weighted_factor_columns].sum(axis=1)

    # For cases where actual value is 0, FAI is equal to the sum of weights for forecasts which are not equal to 0
    actual_is_zero_idx = weighted_fai_df['actual'] == 0
    if np.sum(actual_is_zero_idx) > 0:
        lag_is_not_zero_columns = [f'lag_{x}_is_not_zero' for x in range(1, len(weighting_factors) + 1)]
        weighted_fai_df[lag_is_not_zero_columns] = weighted_fai_df[lag_columns]\
            .apply(lambda x: x != 0, axis=0)\
            .apply(lambda x: x.astype(int), axis=0)
        weighted_fai_df.loc[actual_is_zero_idx, 'fai'] = \
            (weighted_fai_df.loc[actual_is_zero_idx, lag_is_not_zero_columns] * weighting_factors).sum(axis=1)

    # Returning average FAI over all timestamps
    average_fai = weighted_fai_df['fai'].mean()

    return average_fai

if __name__ == "__main__":

    # Getting sample data from darts
    from darts.datasets import AirPassengersDataset, MonthlyMilkDataset

    ts_data = AirPassengersDataset().load()

    # Creating a sequence of dummy predictions (12 lags) and storing them in a sequence of `TimeSeries` instances
    n_lags = 12
    dummy_pred_list = []
    for lag in range(0, n_lags):
        end_idx = len(ts_data) - lag
        dummy_pred_list.append(ts_data[-12-lag:end_idx] + np.random.randint(5))

    # Passing them to an original darts metric (for testing purposes)
    # Note: For original darts metrics, both actual_series and pred_series must have the same length.
    from darts.metrics import mae, mse, rmse
    mae([ts_data for x in dummy_pred_list], dummy_pred_list)
    mse([ts_data for x in dummy_pred_list], dummy_pred_list)
    rmse([ts_data for x in dummy_pred_list], dummy_pred_list)

    # Testing on custom darts metrics
    fai_weights = [0, 0.5, 0.3, 0.2]
    fai_value = fai_dfu(ts_data, dummy_pred_list, weighting_factors=fai_weights, ignore_incomplete_lags=False)
    print(f"Forecast Accuracy Index (FAI) (without ignoring incomplete lags): {fai_value * 100:.2f}%")

    fai_value = fai_dfu(ts_data, dummy_pred_list, weighting_factors=fai_weights, ignore_incomplete_lags=True)
    print(f"Forecast Accuracy Index (FAI) (ignoring incomplete lags): {fai_value * 100:.2f}%")



