# -*- coding: utf-8 -*-
"""
Created on 04-Sep-2020

@author: manuel.blanco.fraga
"""

import pandas as pd
import numpy as np
import datetime
from joblib import Parallel, delayed

from dateutil.relativedelta import relativedelta
from get_model import *

import logging
logger = logging.getLogger()
logger.disabled = True


def prepare_data(demand: pd.DataFrame, df_seg: pd.DataFrame, df_dfu: pd.DataFrame, forecast_level: str = "DFU", target_var: str = "cleansed_demand_volume"):
    """
    Select and clean df to use as input for the time series models.
    Args:
        demand: Demand dataframe
        df_seg: Segmentation dataframe
        df_dfu: DFU mapping to its components
        forecast_level_code: column at which time series models are applied

    Returns:
        Demand dataframe at the forecast level to forecast it,
        demand at the dfu level for disaggregating later if forecast level is different than dfu,
        segmentation, scope to forecast at the forecast level, dfu mapping, original input demand
    """
    
    demand_orig = demand.copy()
    dfu_build = df_dfu.copy()

    demand_dfu = None
    scope = None

    
    df_sales = df_dfu.merge(df_seg, on='DFU')
    lifecycle_cat = df_sales['lifecycle_category'].to_list()
    
    not_obsolete = len(set(lifecycle_cat).intersection(["mature"])) > 0

    if not_obsolete:
        # select segmented using existing DFU and not obsolete
        df_seg = df_seg.loc[((df_seg["in_scope"] == True) | (df_seg["in_scope"] == "True")) & (~df_seg.lifecycle_category.isin(["no_sales", "obsolete"]))]
        scope = df_dfu.loc[df_dfu.DFU.isin(df_seg.DFU), forecast_level].drop_duplicates()
        
        # Filter and aggregate to the right level
        if forecast_level != "DFU":
            demand = pd.merge(demand, df_dfu[["DFU", forecast_level]].drop_duplicates(), on="DFU", how="left")

        demand_dfu = demand.copy()
        demand = demand.loc[demand[forecast_level].isin(scope)]
        demand = demand[[forecast_level, "demand_period", target_var]]
        demand = demand.groupby([forecast_level, "demand_period"]).agg({target_var: 'sum'}).reset_index()
        
        # Set negative demand to zero
        demand.loc[demand[target_var] < 0, target_var] = 0
        demand_dfu.loc[demand_dfu[target_var] < 0, target_var] = 0
        
        # Format date
        demand["demand_period_new"] = pd.to_datetime(demand["demand_period"].astype(str) + '-1', format="%Y-%m-%d")
    else:
        print("All demand is obsolete!")
        demand = pd.DataFrame()

    return demand, demand_dfu, df_seg, scope, dfu_build, demand_orig


def define_snapshots(df: pd.DataFrame, backtest_periods: int, lag: int):
    """
    Define forecast snapshots based on input configuration parameters
    Args:
        df: demand dataframe
        backtest_periods: number of periods to perform backtesting, if 0 or false is future forecast
        lag: lag at which the model competition is evaluated (to leave lag months from last month)

    Returns:
        list of snapshots where te forecast function is executed
    """

    # check if backtest_period is a number or is set to False
    if backtest_periods in [0, False, 'False']:
        backtest = False
    else:
        backtest_periods = int(backtest_periods)
        backtest = True

    demand_periods = df.demand_period.unique()

    # define datetime periods in the proper format
    demand_periods = pd.to_datetime(demand_periods + "-01", format="%Y-%m-%d")
    
    # sort datetime
    demand_periods = demand_periods.sort_values()

    # define snapshots of times
    if not backtest:
        time = [item + relativedelta(months=1) for item in demand_periods[-1:]]
    else:
        time = [item + relativedelta(months=1) for item in demand_periods[-backtest_periods - lag - 1: -lag - 1]]

    return time


def run_time_series_models(demand: pd.DataFrame, models: list, models_params: list, snapshots: list, forecast_level: str = "DFU", lag: int = 1, competition_window: int = 12, target_var: str = "cleansed_demand_volume", horizon: int = 12, backtest_periods = 16):
    """
    Run forecast model for time series, clustering dfus based on their history length and using a
    different approach in each case. A model competition is ran between all the input models to select
    the best performing model for a given snapshot.
    Args:
        demand: demand dataframe
        models: list of models to apply
        models_params: parameters for each individual time series model
        snapshots: snapshots to which this function is applied
        forecast_level: hierarchical level to calculate forecast
        lag: lag at which model competition will be evaluated
        competition_window: window where TS series compete against each other
        target_var: variable to forecast
        horizon: number of periods to forecast
        backtest_periods: periods to BT, if zero or false it calculates a future forecast

    Returns:
        forecast dataframe comprising the forecast for the selected scope for all of the indicated snapshots
    """
    # initializing parameters
    forecast_level = forecast_level
    lag = int(lag)
    competition_window = int(competition_window)
    target_var = target_var

    if backtest_periods in [0, False, 'False', 'false']:
        backtest_period = 0
    else:
        backtest_period = int(backtest_periods)

    # Run forecasting loop
    forecast_list = []
    output_period = 12

    for date in snapshots:
        print("Starting snapshot " + str(date))

        # Select dataframe to forecast
        temp_demand = demand.loc[demand.demand_period_new < date]

        # Separate in groups regarding TS length
        dfu_lengths = temp_demand.groupby(forecast_level)["demand_period"].count()
        
        # Less than frequency
        group_1 = dfu_lengths.loc[(dfu_lengths <= (output_period + lag)) & (dfu_lengths > 3)].index

        # More than frequency but less than competition window
        group_2 = dfu_lengths.loc[
            (dfu_lengths > (output_period + lag)) & (dfu_lengths < (output_period + lag + competition_window))].index

        # More than frequency + competition window
        group_3 = dfu_lengths.loc[dfu_lengths >= (output_period + lag + competition_window)].index

        # Forecasts for each group

        # group1: only short series models
        uniqueDFUs = group_1
        group_1_list = []
        group_1_list = Parallel(n_jobs=8)(delayed(forecast)(x=(temp_demand.loc[temp_demand[forecast_level] == DFU, ["demand_period_new", target_var]]), periodicity=output_period, backtest_periods=backtest_period, date=date, horizon=horizon, lag=lag, competition_window=competition_window, dfu=DFU, target_var=target_var, models=["meanSeries"], params_model=[[len((temp_demand.loc[temp_demand[forecast_level] == DFU, ["demand_period_new", target_var]]).demand_period_new.unique()) - lag]], group=group_1_list) for DFU in uniqueDFUs)
        if group_1_list:
            group_1_list = [item for sublist in group_1_list for item in sublist]
            group_1_fcst = pd.concat(group_1_list)
        else:
            group_1_fcst = pd.DataFrame()
        
        # group2: competition with short window
        uniqueDFUs = group_2
        group_2_list = []
        group_2_list = Parallel(n_jobs=8)(delayed(forecast)(x=(temp_demand.loc[temp_demand[forecast_level] == DFU, ["demand_period_new", target_var]]), periodicity=output_period, backtest_periods=backtest_period, date=date, horizon=horizon, lag=lag, competition_window= (len((temp_demand.loc[temp_demand[forecast_level] == DFU, ["demand_period_new", target_var]]).demand_period_new.unique()) - output_period - lag + 1), dfu=DFU, target_var=target_var,  models=models, params_model=models_params, group=group_2_list) for DFU in uniqueDFUs)
        if group_2_list:
            group_2_list = [item for sublist in group_2_list for item in sublist]
            group_2_fcst = pd.concat(group_2_list)
        else:
            group_2_fcst = pd.DataFrame()
        
        
        # group3: full competition
        uniqueDFUs = group_3
        group_3_list = []
        group_3_list = Parallel(n_jobs=8)(delayed(forecast)(x=(temp_demand.loc[temp_demand[forecast_level] == DFU, ["demand_period_new", target_var]]), periodicity=output_period, backtest_periods=backtest_period, date=date, horizon=horizon, lag=lag, competition_window=competition_window, dfu=DFU, target_var=target_var, models=models, params_model=models_params, group=group_3_list) for DFU in uniqueDFUs)
        if group_3_list:
            group_3_list = [item for sublist in group_3_list for item in sublist]
            group_3_fcst = pd.concat(group_3_list)
        else:
            group_3_fcst = pd.DataFrame()

        finalForecast = pd.concat([group_1_fcst, group_2_fcst, group_3_fcst], axis=0)
        forecast_list.append(finalForecast)
        
    if len(forecast_list)>0:
        forecast_df = pd.concat(forecast_list)
        # Put the correct name
        forecast_df.rename(columns={"DFU": forecast_level}, inplace=True)
    else:
        forecast_df = pd.DataFrame()

    # Remove duplicates (there should not be)
    forecast_df = forecast_df.drop_duplicates([forecast_level, "forecast_date", "forecast_period"])
    forecast_df.reset_index(drop=True, inplace=True)
    return forecast_df


def forecast(x: pd.DataFrame, periodicity: int, backtest_periods: int, horizon: int, lag: int, competition_window: int,
             dfu: str, date: datetime.datetime, target_var: str = "cleansed_demand_volume",
             models: list = ["meanSeries"], params_model: list = [[]], group: list = []):
    """
    Forecast function where each individual model is computed for a time series and a model
    competition is ran.
    Args:
        x: list of values to forecast coming from time series
        periodicity: forecast periodicity in number
        backtest_periods: number of backtest periods
        horizon: number of periods to predict
        date: snapshot date to run the forecast
        lag: lag used to compare models in model competition
        competition_window: number of periods where individual models compete against each other
        dfu: DFU (or equivalent at a given forecast level) being currently forecasted
        target_var: column being forecasted
        models: list of models to participate in the competition
        params_model: list of lists indicating the parameters (if needed) for each model

    Returns:
        dataframe containing the forecast for the time series
    """
    n_periodicity = periodicity

    if len(models) > 1:
        # Model training and competition
        TR_matrix = pd.DataFrame({"demand_period": x[-competition_window:]["demand_period_new"],
                                  "demand": x[-competition_window:][target_var]})

        for newDate in TR_matrix.demand_period.unique():
            newDate_formatted = pd.Timestamp(newDate) - relativedelta(months=lag)

            for i in range(len(models)):
                pred = get_model(x=x.loc[x.demand_period_new < newDate_formatted, target_var],
                                 model=models[i], horizon=lag,
                                 period=n_periodicity, model_params=params_model[i], dates=x.loc[x.demand_period_new < newDate_formatted, "demand_period_new"])
                if np.all(pred == None):
                    TR_matrix.loc[TR_matrix.demand_period == newDate, models[i]] = np.nan
                else:
                    TR_matrix.loc[TR_matrix.demand_period == newDate, models[i]] = pred[-1]

        # Evaluate which model has won
        TR_acc = TR_matrix.drop("demand_period", axis=1)
        for i in range(len(models)):
            TR_acc[models[i]] = abs(TR_acc[models[i]] - TR_acc["demand"])
        D = TR_acc.demand.sum()
        TR_acc = TR_acc.drop("demand", axis=1)
        TR_acc = TR_acc.agg(lambda x: 1 - np.nansum(x) / D, axis=0)

        # Fix NAs
        TR_acc[TR_acc.isna()] = -9999

        bestModel = TR_acc.idxmax()
        TR_acc = TR_acc.reset_index().drop("index", axis=1)
        bestModel_index = TR_acc.idxmax().tolist()[0]
    else:
        bestModel = models[0]
        bestModel_index = 0

    # Calculate forecast

    pred = get_model(x=x[target_var], model=bestModel, horizon=horizon,
                     period=n_periodicity, model_params=params_model[bestModel_index], dates=x["demand_period_new"])

    # Define forecast type
    fcst_type = 'backtest'
    if backtest_periods in [0, False, 'False']:
        fcst_type = 'forecast'
        
    rangeDates = pd.date_range(start=date, end=(date + relativedelta(months=horizon)), freq="MS")
    fcst_dates = rangeDates.strftime("%Y-%m").unique()
    snapshot = date.strftime("%Y-%m")
    fcst_lags = (rangeDates.year - date.year) * n_periodicity + (rangeDates.month - date.month)
    
    tempForecast = pd.DataFrame({"DFU": [dfu] * len(pred),
                                 "forecast_date": [snapshot] * len(pred),
                                 "forecast_period": fcst_dates.to_list(),
                                 "forecast_lag": fcst_lags,
                                 "quantity": pred,
                                 "type": [fcst_type] * len(pred),
                                 "model": [bestModel] * len(pred),
                                 "calculation_date": [pd.to_datetime(datetime.date.today())] * len(pred)})
    
    group.append(tempForecast)

    return group

def disag_fcst(df_agg: pd.DataFrame, demand_agg: pd.DataFrame, demand_disag: pd.DataFrame, df_seg: pd.DataFrame,
               forecast_level: str, target_var: str = "demand_volume", target_level: str = "DFU"):
    """
    Function to disaggregate a forecast to a certain level in the hierarchy (typically DFU), using for this the
    historical demand patterns to calculate the weights applied to each period. It uses as an input a forecast
    aggregated at a higher level than the one desired as input.

    Args:
        demand_agg: Demand aggregated to the level at which the input forecast has been calculated
        df_agg: Input forecast, aggregated at the forecast_level level, different than the target_level
        demand_disag: Demand at the level at which the output forecast is desired
        df_seg: Segmentation dataframe
        forecast_level: Level at which the original forecast was calculated
        target_var: Demand variable to employ in the disaggregation process
        target_level: Level at which the output forecast is desired, usually the DFU level

    Returns: A forecast, in the same format as the input aggregated forecast, but disaggregated to the target level

    """

    demand_agg_bk = demand_agg.copy()
    demand_disag_bk = demand_disag.copy()

    # Set final cols for export
    cols = df_agg.columns.to_list()
    cols.remove(forecast_level)
    cols.append("DFU")
    
    # select segmented using existing DFU and not obsolete
    scope = df_seg.loc[(df_seg["in_scope"] == True) & (~df_seg.lifecycle_category.isin(["no_sales", "obsolete"])), "DFU"].drop_duplicates()
    
    periodicity = df_agg["forecast_period"].apply(lambda x: x.split('-')[1])

    df_agg["P"] = periodicity
    date_list = df_agg.forecast_date.unique().tolist()
    df_disag_final_list = []
    for date in date_list:
        print("Disaggregating for snapshot: " + date)
        # Filter prior to snapshot
        demand_agg_date = demand_agg_bk[demand_agg_bk.demand_period < date]
        demand_disag_date = demand_disag_bk[demand_disag_bk.demand_period < date]

        # Calculate average yearly and period volumes for both agg and disagg demand
        demand_agg_date = create_period_df(demand=demand_agg_date, target_var=target_var, group_by_var=forecast_level)
        demand_disag_date = create_period_df(demand=demand_disag_date, target_var=target_var, group_by_var=target_level)

        # Merge dataframes to obtain final weights of targetLevel-period wrt forecastLevel-period
        weights = pd.merge(demand_agg_date[[forecast_level, "P", "Y_vol", "P_vol"]].drop_duplicates(),
                           demand_disag_date[[target_level, forecast_level, "P", "Y_vol", "P_vol"]].drop_duplicates(),
                           on=[forecast_level, "P"], how="left").dropna()

        # Calculate yearly and period weights
        weights["weight_Y"] = weights["Y_vol_y"] / weights["Y_vol_x"]
        weights["weight_P"] = weights["P_vol_y"] / weights["P_vol_x"]

        weights = weights[[target_level, forecast_level, "P", "weight_Y", "weight_P"]].drop_duplicates()

        # Obtain final weight and normalize it
        weights["weight"] = weights[["weight_Y", "weight_P"]].mean(axis=1)

        # Select only relevant columns
        weights = weights[[forecast_level, target_level, "P", "weight"]]

        # Correct nan weights
        weights = weights.pivot_table("weight", [target_level, forecast_level], "P").reset_index()
        weights = weights.melt(id_vars=[target_level, forecast_level], var_name="P", value_name='weight')
        weights.loc[weights.weight.isna(), "weight"] = 0
        weights = weights.sort_values([forecast_level, target_level, "P"])

        # Merge with aggregated forecast
        df_disag = pd.merge(df_agg[df_agg.forecast_date == date], weights, on=[forecast_level, "P"], how="left")

        # Calculate forecast quantity
        df_disag["quantity"] = df_disag["quantity"] * df_disag["weight"]

        df_disag_final_list.append(df_disag[cols])

    df_disag_final = pd.concat(df_disag_final_list)
    df_disag_final.dropna(inplace=True)
    
    df_disag_final = df_disag_final.loc[df_disag_final["DFU"].isin(scope)]
    
    # Export
    df_disag_final.sort_values(["DFU", "forecast_date", "forecast_period"], inplace=True)
    return df_disag_final

def create_period_df(demand: pd.DataFrame, group_by_var: str, target_var: str):
    """
    Create yearly and period volumes fore each dfu, grouped by year and period.
    It assumes dates come in the YYYY-XX format.
    Args:
        demand: demand dataframe
        group_by_var: unique identifier column
        target_var: demand column to compute averaged

    Returns:
        new demand with new columns added
    """
    demand = demand.copy()
    year = demand["demand_period"].apply(lambda x: x.split('-')[0])
    periodicity = demand["demand_period"].apply(lambda x: x.split('-')[1])

    demand['Y'] = year
    demand["Y_vol"] = demand.groupby([group_by_var, "Y"])[target_var].transform("sum")
    demand["Y_vol"] = demand.groupby([group_by_var])[target_var].transform("mean")

    demand.loc[periodicity.index, 'P'] = periodicity.values

    demand["P_vol"] = demand.groupby([group_by_var, "P"])[target_var].transform("mean")
    return demand
