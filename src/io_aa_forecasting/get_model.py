# -*- coding: utf-8 -*-
"""
Created on 04-Sep-2020

@author: manuel.blanco.fraga
"""

import numpy as np
import pandas as pd

from pmdarima.arima import auto_arima
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from croston import croston
import multiprocessing as mp
import ast
import random

import warnings
import itertools
import statsmodels.api as sm
from fbprophet import Prophet
#from prophet import Prophet

import logging
logger = logging.getLogger()
logger.disabled = True


def get_model(x: pd.Series, model: str, horizon: int, period: int, dates: pd.Series, model_params: list = []):
    """
    Function to retrieve the result from each individual model.
    Args:
        x: list with the values in the time series being forecasted
        model: individual model to call
        horizon: number of periods to forecast
        period: periodicity in numeric format (12 - monthly, 52 - weekly...)
        model_params: list of parameters applicable to the selected model

    Returns:
        list containing the forecast values
    """
    try:
        if model == "meanSeries":
            return mean_series(x, horizon, model_params[0])
        elif model == "naiveSeasonal":
            return naive_seasonal(x, horizon, period)
        elif model == "naiveSeasonalMean":
            return naive_seasonal_mean(x, horizon, period, model_params[0])
        elif model == "arima":
            return arima(x, horizon, period, model_params[0])
        elif model == "newArima":
            return new_arima(x, horizon, period, model_params[0])
        elif model == "ExpSmooth":
            return exp_smooth(x, horizon)
        elif model == "SimpleExpSmooth":
            return simple_exp_smooth(x, horizon)
        elif model == "HoltWinters":
            return holt_winters(x, horizon)
        elif model == "croston":
            return croston_2(x, horizon)
        elif model == "seasonalAverage":
            return seasonal_average(x, horizon, period)
        elif model == "prophet":
            return prophet(x, horizon, dates)
    except Exception as e:
        # print('model ' + model + 'error. Exception ' + str(e))
        return [np.nan] * (horizon + 1)


    

# Individual models
def mean_series(x, horizon, window):
    """
    Calculate mean series model.
    Args:
        x: list with the values in the time series being forecasted
        horizon: number of periods to forecast
        window: number of periods over which average is calculated

    Returns:
        list containing the forecast values
    """
    val = x[-window:].mean()
    val = [val] * (horizon + 1)
    return val


def naive_seasonal(x, horizon, per):
    """
    Calculate naive seasonal model.
    Args:
        x: list with the values in the time series being forecasted
        horizon: number of periods to forecast
        per: periodicity in numeric format (12 - monthly, 52 - weekly...)

    Returns:
        list containing the forecast values
    """
    if len(x) < per:
        raise ValueError("Need at least one year of history")
    else:
        val = x[-per:]
        for i in range(int(np.ceil((horizon + 1) / per))):
            val = val.append(val)
        val = val[:(horizon + 1)].tolist()
        return val


def naive_seasonal_mean(x, horizon, per, window):
    """
    Calculate naive seasonal averaged with a mean model.
    Args:
        x: list with the values in the time series being forecasted
        horizon: number of periods to forecast
        per: periodicity in numeric format (12 - monthly, 52 - weekly...)
        window: number of periods over which average is calculated

    Returns:
        list containing the forecast values
    """
    if len(x) < per:
        raise ValueError("Need at least one year of history")
    else:
        val = x[-per:]
        for i in range(int(np.ceil((horizon + 1) / per))):
            val = val.append(val)
        val = val[:(horizon + 1)].tolist()

        val2 = x[-window:].mean()
        val2 = [val2] * (horizon + 1)

        val3 = [(g + h) / 2 for g, h in zip(val, val2)]
        return val3


def arima(x, horizon, per, seasonal: bool = False):
    """
    Calculate auto arima model.
    Args:
        x: list with the values in the time series being forecasted
        horizon: number of periods to forecast
        per: periodicity in numeric format (12 - monthly, 52 - weekly...)
        seasonal: if true, seasonal arima models will be employed

    Returns:
        list containing the forecast values
    """
    if type(seasonal) == str:
        seasonal = ast.literal_eval(seasonal)
    if len(x) < per:
        raise ValueError("Need at least one year of history")
    else:
        if not seasonal:
            stepwise_model = auto_arima(x, start_p=1, max_p=4,
                                        start_q=1, max_q=2,
                                        start_Q=1, max_Q=2,
                                        start_P=0, max_P=2,
                                        d=0, max_d=0,
                                        D=0, max_D=0,
                                        m=1, seasonal=False,
                                        max_order=5,
                                        trace=False,
                                        error_action='ignore',
                                        suppress_warnings=True,
                                        stepwise=False,
                                        n_jobs=mp.cpu_count())
        else:
            stepwise_model = auto_arima(x, start_p=1, max_p=4,
                                        start_q=1, max_q=2,
                                        start_Q=1, max_Q=2,
                                        start_P=0, max_P=2,
                                        d=0, max_d=0,
                                        D=0, max_D=0,
                                        m=per, seasonal=True,
                                        max_order=5,
                                        trace=False,
                                        error_action='ignore',
                                        suppress_warnings=True,
                                        stepwise=False,
                                        n_jobs=mp.cpu_count())

        stepwise_model.fit(x)
        result = stepwise_model.predict(n_periods=int(horizon + 1))
        return result


def exp_smooth(x, horizon):
    """
    Calculate exponential smoothing.
    Args:
        x: list with the values in the time series being forecasted
        horizon: number of periods to forecast

    Returns:
        list containing the forecast values
    """
    M = ExponentialSmoothing(x[x > 0]).fit(use_boxcox=True)
    return M.forecast(int(horizon + 1)).to_list()


def simple_exp_smooth(x, horizon):
    """
    Calculate simple exponential smooting.
    Args:
        x: list with the values in the time series being forecasted
        horizon: number of periods to forecast

    Returns:
        list containing the forecast values
    """
    M = SimpleExpSmoothing(x).fit()
    return M.forecast(int(horizon + 1)).to_list()


def holt_winters(x, horizon):
    """
    Holt winters model.
    Args:
        x: list with the values in the time series being forecasted
        horizon: number of periods to forecast

    Returns:
        list containing the forecast values
    """
    M = Holt(x).fit()
    return M.forecast(horizon + 1).to_list()


def croston_1(x, horizon, alpha=0.4):
    """
    Croston model.
    Args:
        x: list with the values in the time series being forecasted
        horizon: number of periods to forecast
        alpha: smoothing constant

    Returns:
        list containing the forecast values
    """
    d = x[:]
    cols = len(d)  # Historical period length
    d = np.append(d, [np.nan] * horizon)  # Append np.nan into the demand array to cover future periods

    # level (a), periodicity(p) and forecast (f)
    a, p, f = np.full((3, cols + horizon), np.nan)
    q = 1  # periods since last demand observation

    # Initialization
    first_occurence = np.argmax(d[:cols] > 0)
    a[0] = d[first_occurence]
    p[0] = 1 + first_occurence
    f[0] = a[0] / p[0]
    # Create all the t+1 forecasts
    for t in range(0, cols):
        if d[t] > 0:
            a[t + 1] = alpha * d[t] + (1 - alpha) * a[t]
            p[t + 1] = alpha * q + (1 - alpha) * p[t]
            f[t + 1] = a[t + 1] / p[t + 1]
            q = 1
        else:
            a[t + 1] = a[t]
            p[t + 1] = p[t]
            f[t + 1] = f[t]
            q += 1

    # Future Forecast
    a[cols + 1:cols + horizon] = a[cols]
    p[cols + 1:cols + horizon] = p[cols]
    f[cols + 1:cols + horizon] = f[cols]

    return f[-horizon:]


def croston_2(x, horizon):
    """
    Preimplemented croston model.
    Args:
        x: list with the values in the time series being forecasted
        horizon: number of periods to forecast

    Returns:
        list containing the forecast values
    """
    return croston.fit_croston(x, int(horizon + 1))["croston_forecast"]


def seasonal_average(x, horizon, per):
    """
    Seasonal average model, calculating the forecast as the average of the last years for each period.
    Args:
        x: list with the values in the time series being forecasted
        horizon: number of periods to forecast
        per: periodicity in numeric format (12 - monthly, 52 - weekly...)

    Returns:
        list containing the forecast values
    """
    vals = [np.nan] * (horizon + 1)
    nLags = int(np.floor(len(x) / per))
    x = x.to_list()

    if len(x) < per:
        raise ValueError("Need at least one year of history")
    else:
        for h in range(horizon + 1):
            if (h - per) < 0:
                vals[h] = x[-per + h]
            else:
                vals[h] = vals[-per + h]
            for i in range(nLags - 1):
                if (h - (per * (i + 2))) < 0:
                    vals[h] += x[-int(per * (i + 2)) + h]
                else:
                    vals[h] += vals[-int(per * (i + 2)) + h]
            vals[h] = vals[h] / nLags

    return vals

def new_arima(x, horizon, per, optimize = True):
    """
    Calculate auto seasonal arima model.
    Args:
        x: list with the values in the time series being forecasted
        horizon: number of periods to forecast
        per: periodicity in numeric format (12 - monthly, 52 - weekly...)
        seasonal: if true, seasonal arima models will be employed

    Returns:
        list containing the forecast values
    """
    if len(x) < per:
        raise ValueError("Need at least one year of history")
    else:
        # Define the p, d and q parameters to take any value between 0 and 2
        p = d = q = range(0, 2)

        # Generate all different combinations of p, q and q triplets
        pdq = list(itertools.product(p, d, q))

        # Generate all different combinations of seasonal p, q and q triplets
        seasonal_pdq = [(k[0], k[1], k[2], per) for k in list(itertools.product(p, d, q))]

        # Calculate the optimal parameters+
        if (optimize == True):
            minAIC = np.inf
            optimalPDQ = (1,1,1)
            optimalSeasPDQ = (1,1,1,12)
            for param in pdq:
                for param_seasonal in seasonal_pdq:
                    try:
                        mod = sm.tsa.statespace.SARIMAX(x,
                                                        order=param,
                                                        seasonal_order=param_seasonal,
                                                        enforce_stationarity=False,
                                                        enforce_invertibility=False)

                        results = mod.fit()
                        # Compare for discarding zero cases
                        F = results.predict(0, horizon).sum()
                        A = np.sum(x[-horizon:])

                        if ((results.aic < minAIC) and (F / A > 0.5) and (F / A) < 2.0):
                            minAIC = results.aic
                            optimalPDQ = param
                            optimalSeasPDQ = param_seasonal
#                             print(F / A)
#                             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                    except:
                        continue
        else:
            optimalPDQ = (1,1,1)
            optimalSeasPDQ = (1,1,1,12)
        # Make predictions
        mod = sm.tsa.statespace.SARIMAX(x,
                                    order=optimalPDQ,
                                    seasonal_order=optimalSeasPDQ,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
        result = mod.fit()
        result = result.predict(0, horizon)
        
        return result
    
def prophet(x, horizon, dates):
    """
    Facebook's Prophet model.
    Args:
        x: list with the values in the time series being forecasted
        horizon: number of periods to forecast
        dates: list of dates to inject the model

    Returns:
        list containing the forecast values
    """
    
    # Initialize dataframe
    df = pd.DataFrame({"ds": dates, "y": x})
    
    # Fit model
    m = Prophet().fit(df)
    
    # Make future prediction filtering out irrelevant dates
    future = m.make_future_dataframe(periods=(horizon + 2),freq='M')
    future = future.loc[future["ds"] > max(dates)]
    future = future.iloc[1:]
    forecast = m.predict(future)
    
    return forecast['yhat'].tolist()