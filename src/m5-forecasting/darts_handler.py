"""
Supporting functions for forecasting using the Darts framework.
"""

from typing import List, Dict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.metrics import mae, mape, rmse

from forecast_metrics import fai_wts_dfu
from .base import loguru_logger, log_time


class DartsHandler:
    """
    A class for handling the model training, evaluation, and backtesting using the Darts framework.
    """
    def __init__(self, weighting_factors: List[float], forecast_horizon: int, datetime_var: str, 
                 target_var: str, x_vars: List[str]=None, dfu_name: str=''):

        self.created_date = datetime.now()
        self.weighting_factors = weighting_factors
        self.forecast_horizon = forecast_horizon
        self.datetime_var = datetime_var
        self.target_var = target_var
        self.x_vars = x_vars
        self.dfu_name = dfu_name
        self.model = None
        self.backtest_results = None
        self._logger = loguru_logger

    @log_time
    def fit_and_evaluate(self, model, data_df: pd.DataFrame, train_percent: float, 
                         plot: bool=False, backtest: bool=False, **kwargs):
        """
        Fits the Darts model to the training TimeSeries and evalautes it on the validation TimeSeries.

        Args:
            model: Models which are compatible with Darts framework.
            data_df (pd.DataFrame): DataFrame containing the target variable and the x_vars (if provided).
            train_percent (float): The percentage of the data to be used for training. 
            plot (bool, optional): [description]. Defaults to False.
            backtest (bool, optional): [description]. Defaults to False.
        """

        # =============================================================================
        # DATA PREPROCESSING
        # =============================================================================
        # Splitting input data into Train & Validation TimeSeries
        data_df = self.train_validation_split(
            data_df=data_df, train_percent=train_percent, datetime_var=self.datetime_var, target_var=self.target_var, 
            x_vars=self.x_vars
        )

        train_df = data_df.loc[data_df['DataGroup'] == 'Train', :].copy()
        validation_df = data_df.loc[data_df['DataGroup'] == 'Validation', :].copy()

        train_dict = self.pd_to_timeseries(train_df, self.datetime_var, self.target_var, self.x_vars)
        validation_dict = self.pd_to_timeseries(validation_df, self.datetime_var, self.target_var, self.x_vars)

        train_ts_target = train_dict['target']
        validation_ts_target = validation_dict['target']
        train_ts_x_vars = train_dict['x_vars']

        # Trending the Train/Validation portion of the data
        if plot:
            fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
            train_ts_target.plot(ax=ax, label='Train')
            validation_ts_target.plot(ax=ax, label='Validation')
            ax.set_title(f"{self.dfu_name}")
            fig.show()

        # =============================================================================
        # MODEL FITTING
        # =============================================================================
        # Fitting the model
        if self.x_vars is None:
            self.model = model.fit(train_ts_target)
        else:
            self.model = model.fit(train_ts_target, past_covariates=train_ts_x_vars)

        # Forecasting the validation portion of the data
        pred_validation_ts = self.model.predict(n=len(validation_ts_target))

        # Calculating metrics
        # Note: FAI & WTS are not available here because FAI & WTS requires backtesting
        validation_mae = mae(validation_ts_target, pred_validation_ts)
        validation_mape = mape(validation_ts_target, pred_validation_ts)
        validation_rmse = rmse(validation_ts_target, pred_validation_ts)

        # Plotting actual vs forecasted values
        if plot:
            fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
            train_ts_target.plot(ax=ax, label='Train')
            validation_ts_target.plot(ax=ax, label='Validation')
            self.model.plot(ax=ax, label='Forecast (AutoARIMA)')
            ax.set_title(f"{self.dfu_name}\nMAE: {validation_mae:.2f} | MAPE: {validation_mape:.2f}% | " \
                         f"RMSE: {validation_rmse:.2f}")
            fig.show()

        # =============================================================================
        # BACKTESTING
        # =============================================================================
        if backtest:

            # Using the entire dataset for backtesting
            backtest_start_time = data_df[self.datetime_var].values[self.model.min_train_series_length]
            self.backtest_results = self.run_backtest(
                data_df=data_df, start=backtest_start_time, plot=plot
            )
            fai_score, wts_score = self.backtest_results['fai'], self.backtest_results['wts']
            
        else:
            fai_score, wts_score = None, None
            
        return pred_validation_ts, fai_score, wts_score
        
    @log_time
    def run_backtest(self, data_df: pd.DataFrame, start: datetime=None, retrain: bool=True, plot: bool=False, 
                     **kwargs):
        """
        Runs a backtest using the model fitted, given the data and start date.

        Args:
            data_df (pd.DataFrame): DataFrame to be backtested on.
            start (datetime, optional): Starting time at which backtest begins. Defaults to None, where the start time
                is determined based on model.min_train_series_length.
            retrain (bool, optional): Whether to retrain model on every backtest iteration. It is recommended to turn 
                this off (False) if model training takes a long time. Defaults to True.
            plot (bool, optional): Whether to plot results. Defaults to False.

        Returns:
            A dictionary containing the following key-value pairs:
                1. fai: FAI score calculated.
                2. wts: WTS score calculated.
                3. results: A list of TimeSeries objects containing the results of the backtest.
        """

        # First check if model is fitted
        assert self.model is not None, "Model must be fitted before running backtest."

        # Preparing the data for backtesting
        self._logger.debug(f"Preparing data for backtesting.")
        data_dict = self.pd_to_timeseries(data_df, self.datetime_var, self.target_var, self.x_vars)
        data_ts_target = data_dict['target']
        data_ts_x_vars = data_dict['x_vars']

        # Running backtest
        self._logger.debug("Running backtests...")
        historical_forecast = self.model.historical_forecasts(
            series=data_ts_target, 
            past_covariates=data_ts_x_vars if len(data_ts_x_vars) > 0 else None,
            start=start,
            forecast_horizon=self.forecast_horizon,
            last_points_only=False,
            retrain=retrain,
            verbose=True
        )
        self._logger.debug(f"Backtest completed.")

        # FAI & WTS calculation
        fai_score, wts_score = \
            fai_wts_dfu(data_ts_target, historical_forecast, weighting_factors=self.weighting_factors)
        self._logger.info(f"FAI: {fai_score * 100 :.2f}% | WTS: {wts_score:.3f}")

        # Aggregating other backtest metrics
        mae_list, mape_list, rmse_list = [], [], []
        for single_forecast in historical_forecast:
            mae_list.append(mae(single_forecast, data_ts_target.slice_intersect(single_forecast)))
            mape_list.append(mape(single_forecast, data_ts_target.slice_intersect(single_forecast)))
            rmse_list.append(rmse(single_forecast, data_ts_target.slice_intersect(single_forecast)))

        mae_avg, mape_avg, rmse_avg = np.mean(mae_list), np.mean(mape_list), np.mean(rmse_list)
        self._logger.info(f"[Backtest] MAE: {mae_avg:.2f} | MAPE: {mape_avg * 100:.2f}% | RMSE: {rmse_avg:.2f}")

        results_dict = {
            'fai': fai_score,
            'wts': wts_score,
            'results': historical_forecast
        }

        return results_dict
        
    @log_time
    def forecast(self, data_df: pd.DataFrame=None, forecast_horizon: int=None, plot: bool=False, **kwargs):
        """
        Runs a forecast based on the Darts model trained.

        Args:
            data_df (pd.DataFrame, optional): Input DataFrame to be forecasted on. Only required for models with 
                past_covariates (x_vars).
            forecast_horizon (int, optional): Number of steps to forecast. Defaults to None, where the default class 
                value is used.
            plot (bool, optional): Whether to plot results. Defaults to False.

        Returns:
            A TimeSeries object of the forecasted values.
        """

        # Forecasting
        forecast_horizon = forecast_horizon or self.forecast_horizon
        if self.x_vars is None:
            forecast_results = self.model.predict(n=forecast_horizon)

        else:
            # Preparing data_df for forecasting (converting to TimeSeries)
            data_dict = self.pd_to_timeseries(data_df, self.datetime_var, self.target_var, self.x_vars)
            data_ts_x_vars = data_dict['x_vars']

            forecast_results = self.model.predict(n=forecast_horizon, past_covariates=data_ts_x_vars)

        # Rearranging data into a DataFrame
        forecast_df = self.timeseries_to_pd({self.target_var: forecast_results})

        return forecast_df

    @log_time
    @staticmethod
    def pd_to_timeseries(data_df: pd.DataFrame, datetime_var: str, target_var: str, x_vars: List[str]=None):
        """
        Converts a pandas DataFrame into a dictionary of TimeSeries objects.

        Args:
            data_df (pd.DataFrame): DataFrame containing data used for model training.
            datetime_var (str): Name of the column in the DataFrame that contains the datetime information.
            target_var (str): Name of the column in the DataFrame that contains the target variable.
            x_vars (List[str], optional): List of the columns in the DataFrame that are used as covariates. 
                Defaults to None (no covariates).

        Returns:         
            A dictionary containing the following key-value pairs:
                1. target: Target TimeSeries object.
                2. x_vars: List of covaiate TimeSeries objects.
        """

        # Target TimeSeries
        target_ts = TimeSeries.from_dataframe(data_df, datetime_var, target_var)

        # Covariate (x_vars) TimeSeries
        x_vars_ts_list = []
        if x_vars is not None:
            for x_var in x_vars:
                x_var_ts = TimeSeries.from_dataframe(data_df, datetime_var, x_var)
                x_vars_ts_list.append(x_var_ts)

        ts_dict = {
            'target': target_ts,
            'x_vars': x_vars_ts_list
        }

        return ts_dict

    @log_time
    @staticmethod
    def timeseries_to_pd(ts_dict: Dict[TimeSeries]):
        """
        Converts a list of TimeSeries objects into a pandas DataFrame.

        Args:
            ts_dict (Dict[TimeSeries]): Dictionary of TimeSeries objects in the form of 'var_name': TimeSeries.
        """
        
        data_df = None
        for var_name, ts in ts_dict.items():
            
            ts_df = pd.DataFrame({
                self.datetime_var: ts.time_index,
                var_name: ts.values().flatten()
            })

            if data_df is None:
                data_df = ts_df
            else:
                data_df = data_df.merge(ts_df, on=self.datetime_var, how='outer')
                
        return data_df

    @log_time
    @staticmethod
    def train_validation_split(data_df: pd.DataFrame, train_percent: float, datetime_var: str, target_var: str,
                               x_vars: List[str]=None):
        """
        Splits the data into training and validation sets.

        Args:
            data_df (pd.DataFrame): DataFrame containing data used for model training.
            train_percent (float): Percentage of the data to be used for training.
            datetime_var (str): Name of the column in the DataFrame that contains the datetime information.
            target_var (str): Name of the column in the DataFrame that contains the target variable.
            x_vars (List[str], optional): List of the columns in the DataFrame that are used as covariates. 
                Defaults to None (no covariates).
        """

        # Managing if train_percent is provided in the form of 0-100
        if train_percent > 1:
            train_percent = train_percent / 100

        # Getting the end row index for training, remaining would be validation
        train_end_idx = np.ceil(data_df.shape[0] * train_percent)

        # The rows which will be used for Training and Validation are indicated under the DataGroup column
        data_df['DataGroup'] = 'Validation'
        data_df['DataGroup'].iloc[:train_end_idx] = 'Train'

        # Subsetting only relevant columns
        subset_vars = [datetime_var, target_var]
        if x_vars is not None:
            subset_vars += x_vars
        data_df = data_df[subset_vars].copy()

        return data_df



if __name__ == "__main__":

    pass
