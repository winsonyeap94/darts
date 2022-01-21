# -*- coding: utf-8 -*-
"""
Created on 02-Sep-2020

@author: manuel.blanco.fraga
"""

import pandas as pd
import numpy as np

def calculate_lag(forecast, periodicity = "monthly"):
    """
    Calculate forecast lag for monthly forecasts.
    Args:
        forecast: forecast dataframe

    Returns:
        input dataframe containing forecast lags
    """
    
    if periodicity == "monthly":
        p = 12
    elif periodicity == "weekly":
        p = 52
    
    forecast[['f_period_year', 'f_period_month']] = forecast["forecast_period"].str.split('-', expand=True).astype(int)
    forecast[['f_date_year', 'f_date_month']] = forecast["forecast_date"].str.split('-', expand=True).astype(int)
    forecast["forecast_lag"] = (forecast["f_period_year"] - forecast["f_date_year"]) * p + (forecast["f_period_month"] - forecast["f_date_month"])
    forecast = forecast.drop(['f_period_year', 'f_period_month', 'f_date_year', 'f_date_month'], axis=1)
    return forecast

def fillzeros(demand: pd.DataFrame, id_var: str = "DFU", date_var: str = "demand_period", target_var: str = "demand_volume"):
    """
    Fill missing demand values with zeros.
    Args:
        demand: demand dataframe
        id_var: unique identifier
        date_var: dates
        target_var: variable to fill

    Returns:
        demand dataframe with missing values filled
    """
    
    # Pivot table and find first demand
    pivoted = demand.pivot_table(values=target_var, index=id_var, columns=date_var, aggfunc='sum').reset_index()
    cols = pivoted.columns
    pivoted["first_demand"] = pivoted[cols[~cols.str.contains("DFU")]].apply(lambda x: np.min(pivoted[cols[~cols.str.contains(id_var)]].columns[(x > 0) & (~(x.isna()))]), axis=1)
    pivoted = pivoted.loc[~(pivoted["first_demand"].isna())]
    
    # Melt table and set zero for all dates after the first demand
    melted = pivoted.melt(id_vars=[id_var, "first_demand"], var_name=date_var, value_name=target_var)
    melted.loc[(melted[date_var] >= melted["first_demand"]) & (melted[target_var].isna()), target_var] = 0
    
    # Discard the first demand
    melted = melted.drop("first_demand", axis = 1)
    
    return melted

def clean_nan_demand(demand: pd.DataFrame):
    """
    Fill missing demand values with zeros when any of the cleansed or raw are missing.
    Args:
        demand: demand dataframe containing raw and cleansed demand

    Returns:
        demand dataframe with missing values filled according to logic
    """
        # Both are NaN
    demand.loc[demand["demand_volume"].isna() & demand["cleansed_demand_volume"].isna(), ["demand_volume", "cleansed_demand_volume"]] = 0
        # Only raw is NaN
    demand.loc[demand["demand_volume"].isna() & ~demand["cleansed_demand_volume"].isna(), "demand_volume"] = 0
        # Only clean is NaN
    demand.loc[~demand["demand_volume"].isna() & demand["cleansed_demand_volume"].isna(), "cleansed_demand_volume"] = demand.loc[~demand["demand_volume"].isna() & demand["cleansed_demand_volume"].isna(), "demand_volume"]
    
    return demand

