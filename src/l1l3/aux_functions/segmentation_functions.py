# -*- coding: utf-8 -*-
"""
Created on 03-Sep-2020

@author: manuel.blanco.fraga
"""

import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from statsmodels.formula.api import ols

def regression(data, yvar, xvars):
    """
    Function to perform a linear regresson of a dataframe
    :param data:
    :param yvar:
    :param xvars:
    :return:
    """
    y = data[yvar]
    x = data[xvars]
    lm = LinearRegression()
    lm.fit(x, y)
    result = lm.coef_[0]
    # print(result)
    return result


def get_lifecycle_category(row, min_periods_new, max_periods_obsolete):
    """
    Generate lifecycle category
    :param row:
    :param min_periods_new: number of periods to label a dfu as new
    :param max_periods_obsolete: number of periods with zero demand from last date available to consider it obsolete
    :return: categorization of the dfus based on their lifecycle
    """
    if (math.isnan(row['phase_out'])) | (math.isnan(row['phase_in'])):
        lifecycle_cattegory = 'no_sales'
    elif row['phase_out'] <= max_periods_obsolete:
        lifecycle_cattegory = 'obsolete'
    elif row['phase_in'] >= min_periods_new:
        lifecycle_cattegory = 'new_product'
    elif row['grade'] < (-1 / 12):
        lifecycle_cattegory = 'declining'
    else:
        lifecycle_cattegory = 'mature'
    return lifecycle_cattegory


def compute_lifecycle(df, id_col, name_quantity_col, name_date_col, obsolete_threshold=None, start_date=None):
    """
    Function to calculate the lifecycle of the dfus in the dataframe
    :param df: input dataframe containing the demand
    :param id_col: column where the dfu unique identifier is stored
    :param name_quantity_col: column containing the demand volumes
    :param name_date_col: column containing date information
    :param obsolete_threshold: number of periods with zero demand to label the dfu as obsolete
    :param start_date: initial date to do the analysis
    :return: dataframe with the lifecycle of each dfu
    """
    df[name_date_col] = pd.to_datetime(df[name_date_col].apply(lambda x: str(x) + "-1"), format="%Y-%m-%d", errors="ignore")

    if start_date is None:
        start_date = min(df[name_date_col])

    periods = df[name_date_col].unique()
    periods = pd.Series(periods)
    periods = periods[periods >= start_date]
    periods = periods.sort_values(ascending=True)
    index = pd.Series(periods.index).sort_values()
    periods.index = index
    periods = periods.to_dict()
    periods_date_as_key = {y: x for x, y in periods.items()}

    df['date_numeric'] = df[name_date_col].map(periods_date_as_key)

    temp = df[:]
    temp = temp[temp[name_quantity_col] != 0]
    temp['phase_in'] = df[df[name_quantity_col] != 0].groupby(id_col)['date_numeric'].transform('min')
    temp['phase_out'] = df[df[name_quantity_col] != 0].groupby(id_col)['date_numeric'].transform('max')
    temp = temp[[id_col, "phase_in", "phase_out"]].drop_duplicates()
    df = pd.merge(df, temp, right_on=id_col, left_on=id_col, how='left')

    df['sum_qty'] = df.groupby(id_col)[name_quantity_col].transform('sum')
    df['mean_sales'] = df.groupby(id_col)[name_quantity_col].transform('mean')

    df_slope = df.groupby(id_col).apply(regression, name_quantity_col, ['date_numeric'])
    df_slope = df_slope.reset_index()
    df_slope = df_slope.rename(columns={0: 'slope'})

    df = pd.merge(df[[id_col, 'phase_in', 'phase_out', 'sum_qty', 'mean_sales']].drop_duplicates(),
                  df_slope, right_on=id_col, left_on=id_col, how='left')
    df['grade'] = df['slope'] / df['mean_sales']

    limit = df.phase_out.max()

    if obsolete_threshold is None:
        df['lifecycle_category'] = df.apply(lambda x: get_lifecycle_category(x, min_periods_new=limit - 12, max_periods_obsolete=limit - 12), axis=1)
    else:
        df['lifecycle_category'] = df.apply(lambda x: get_lifecycle_category(x, min_periods_new=limit - 12, max_periods_obsolete= limit - obsolete_threshold), axis=1)
    
    lifecycle_df = df[[id_col, 'lifecycle_category', 'phase_in', 'phase_out', 'grade']].drop_duplicates()
    lifecycle_df["phase_out"] = lifecycle_df['phase_out'].map(periods)
    lifecycle_df["phase_in"] = lifecycle_df['phase_in'].map(periods)

    return lifecycle_df



def compute_sum_by_id(df: pd.DataFrame, id_column_name: str, target_column_name: str, date_column_name: str):
    """
    Computes sum of target_column_name by ID in last year and in all period.
    Args:
        df: data frame containing demand information
        id_column_name: name of column inside df containing ID
        target_column_name: name of column inside df containing the target variable
        date_column_name: name of column inside df containing date

    Returns: df

    """
    # sum the target_column by ID
    sums_demand = df.copy()
    sums_demand['sum_qty'] = sums_demand.groupby([id_column_name])[target_column_name].transform('sum')

    # filter by last year and repeat exercise
    temp = sums_demand.copy()
    months = 12
    dates = df[date_column_name].unique()
    dates.sort()
    dates = dates[-months:]
    temp = temp.loc[(temp[date_column_name].isin(dates))]
    temp['last_year_sum_qty'] = temp.groupby(id_column_name)[target_column_name].transform('sum')
    temp = temp[[id_column_name, "last_year_sum_qty"]].drop_duplicates()

    # merge data frames
    sums_demand = pd.merge(sums_demand, temp.drop_duplicates(), right_on=id_column_name, left_on=id_column_name,
                           how='left')
    return sums_demand


def compute_intermitency(x):
    """
    Function to compute the intermittency of a series
    :param x: list containing numerical values of the series
    :return: intermittency
    """
    return np.mean(np.diff(np.where(x > 0)))


def compute_cov(x):
    """
    Code to compute the coefficient of variation of a series
    :param x: list containing numerical values on the series
    :return: coefficient of variation
    """
    return np.std(x) / np.mean(x)


def count_not_null_rows(column):
    """
    Function to count rows that are not null in a column
    :param column: target column to measure
    :return: number of not null rows
    """
    return column.ne(0).sum()


def seasonal_trend_cov(x, periodicity):
    """
    Separation of a series in seasonal and trend components, and calculation of the coefficient of variation with the remainder
    :param x: list of values in the series
    :param periodicity: periodicity of the series
    :return: pandas series indicating if the series analyzed is seasonal, if it has a trend, its cov and a categorization of its seasonal degree
    """
    # filter only series with more than 2 years and at least one not null point each year
    if (len(x) >= 2 * periodicity) & (
            x[(max(0, len(x) - 2 * periodicity)):(len(x) - periodicity)].astype(bool).sum() > 0) & (
            x[len(x) - periodicity:len(x)].astype(bool).sum() > 0):
        try:
            stl = STL(x, period=periodicity)
        except:
            # print('error')
            if x.mean() == 0:
                # print('mean = 0')
                seasonal = 0
                trend = 0
                cov = np.nan
                seasonal_degree = "non_seasonal"
            else:
                # print('mean != 0')
                seasonal = 0
                trend = 0
                cov = compute_cov(x)
                seasonal_degree = "non_seasonal"
        else:
            seasonal = 0
            trend = 0
            seasonal_degree = "non_seasonal"
            # print('success')
            res = stl.fit()
            # print(res)
            if res.trend.mean() == 0:
                # print('trend mean = 0')
                cov = np.nan
            else:
                # print('trend mean != 0')
                # just for last year (12 months)
                L = len(x)
                minL = max(0, L - periodicity)
                cov = compute_cov(res.resid[minL:L] + res.trend[minL:L])
            if x.mean() == 0:
                # print('mean = 0')
                seasonal = 0
                trend = 0
                seasonal_degree = "non_seasonal"
            else:
                # print('mean != 0')
                y = x[periodicity:len(x)]
                z = x[0:len(y)]
                if (len(y) != 0) & (len(z) != 0):
                    # print('len!=0')
                    # df = pd.concat([y.rename('y'), z.rename('z')], axis=1)
                    # print('df done')
                    est = sm.OLS(list(z), y).fit()
                    r_square = est.rsquared_adj
                    # print('ols done')
                    if len(est.params) == 0:
                        # print('len est params = 0')
                        seasonal = 0
                        seasonal_degree = "non_seasonal"
                    else:
                        # print('len est params != 0')
                        p_value = est.pvalues.values[0]
                        if (p_value is None):
                            # print('p value none')
                            seasonal = 0
                            seasonal_degree = "non_seasonal"
                        else:
                            if (p_value <= 0.05):
                                # print('p value <= 0.05')
                                seasonal = 1
                                if (r_square >= 0.8):
                                    # print('rsquare >= 0.08')
                                    seasonal_degree = "high_seasonal"
                                elif (r_square < 0.8) & (r_square >= 0.3):
                                    # print('rsquare >= 0.03')
                                    seasonal_degree = "high_seasonal"
                                else:
                                    # print('rsquare < 0.03')
                                    seasonal_degree = "low_seasonal"
                            else:
                                # print('else')
                                seasonal = 0
                                seasonal_degree = "non_seasonal"
                df1 = x.reset_index(drop=True).rename('x').reset_index()
                # print('df done')
                # print(df1)
                res = ols("x ~ index", df1).fit()
                # print('ols done')
                p_coef = res.pvalues[1]
                if p_coef is None:
                    trend = 0
                else:
                    if (p_coef <= 0.05):
                        trend = np.sign(res.params[1]) * 1
    else:
        trend = 0
        if x.mean() == 0:
            # print('mean = 0')
            seasonal = 0
            trend = 0
            cov = np.nan
            seasonal_degree = "non_seasonal"
        else:
            # print('mean != 0')
            seasonal = 0
            trend = 0
            cov = compute_cov(x)
            seasonal_degree = "non_seasonal"

    return pd.Series({'seasonal': seasonal,
                      'trend': trend,
                      'cov': cov,
                      'seasonal_degree': seasonal_degree})


def compute_segmentation_and_intermittency(df: pd.DataFrame, id_col: str, name_qty_col: str, name_date_col: str, abc_last_year: bool = False):
    """
    Runs ABC & XYZ segmentations and intermittency analysis
    Args:
        df: data frame containing historical sales
        id_col: name of column containing ID/DFU
        name_qty_col: name of column with historical demand
        name_date_col: name of date columns
        abc_last_year: use only last year for ABC segmentation

    Returns:
        Dataframe containing the segmentation features
    """
    results = df[[id_col]].copy().drop_duplicates()

    # compute ABC segmentation
    df_abc = df.copy()

    if abc_last_year:
        months = 12
        dates = df[name_date_col].unique()
        dates.sort()
        dates = dates[-months:]
        df_abc = df_abc.loc[(df_abc[name_date_col].isin(dates))]

    df_abc[name_qty_col] = df_abc.groupby([id_col])[name_qty_col].transform('sum')
    df_abc = df_abc[[id_col, name_qty_col]].copy().drop_duplicates().sort_values(by=name_qty_col,
                                                                                 ascending=False)
    df_abc['cumsum_qty'] = df_abc[name_qty_col].cumsum()
    df_abc['cumperc_qty'] = df_abc['cumsum_qty'] / df_abc[name_qty_col].sum()
    
    df_abc.loc[df_abc['cumperc_qty'] != np.nan, 'ABC'] = 'B'
    df_abc.loc[df_abc['cumperc_qty'] <= 0.8, 'ABC'] = 'A'
    df_abc.loc[df_abc['cumperc_qty'] > 0.95, 'ABC'] = 'C'
    df_abc["ABC"].iloc[0] = 'A'
    
    results = pd.merge(results, df_abc[[id_col, 'ABC']].drop_duplicates(),
                       right_on=[id_col],
                       left_on=[id_col], how='left')
    
    #'----- ABC analysis computed-----'#
    
    # Compute CoV, seasonality and trend
    df_season = df.copy()
    df_season = df_season.groupby([id_col])[name_qty_col].apply(lambda x: seasonal_trend_cov(x, 12))
    df_season = df_season.reset_index()

    new_df = pd.pivot_table(df_season, values=name_qty_col, index=[id_col], columns=['level_1'],
                            aggfunc=np.sum)
    
    del new_df['cov']
    new_df = pd.merge(new_df, df_season[df_season.level_1 == 'cov'][[id_col, name_qty_col]].rename(
        columns={name_qty_col: 'cov'}), left_on=id_col, right_on=id_col, how='left')

    new_df.loc[new_df['cov'] >= 0.5, 'XYZ'] = "Z"
    new_df.loc[(new_df['cov'] < 0.5) & (new_df['cov'] >= 0.3), 'XYZ'] = "Y"
    new_df.loc[new_df['cov'] < 0.3, 'XYZ'] = "X"

    new_df.loc[new_df['cov'] >= 0., 'CV_result'] = "high"
    new_df.loc[new_df['cov'] < 0.5, 'CV_result'] = "low"

    new_df.loc[new_df['trend'] == 0, 'trend_degree'] = "no_trend"
    new_df.loc[new_df['trend'] == 1, 'trend_degree'] = "positive_trend"
    new_df.loc[new_df['trend'] == -1, 'trend_degree'] = "negative_trend"

    new_df.reset_index(inplace=True)
    
    results = pd.merge(results, new_df.drop_duplicates(),
                       right_on=[id_col],
                       left_on=[id_col], how='left')
    
    #'------------CoV, seasonality and trend analysis computed-------------'#

    # Compute intermittency/ADI (avg demand interval)
    
    df_adi = df.copy()
    df_adi['n_periods'] = df_adi.groupby([id_col])[name_date_col].transform('count')
    df_adi['n_not_null_periods'] = df_adi.groupby([id_col])[name_qty_col].transform(count_not_null_rows)
    df_adi['ADI'] = df_adi['n_not_null_periods'] / df_adi['n_periods']

    df_adi.loc[df_adi['ADI'] >= 0.73, 'ADI_result'] = "low"
    df_adi.loc[df_adi['ADI'] < 0.73, 'ADI_result'] = "high"

    results = pd.merge(results, df_adi[[id_col, 'ADI', 'ADI_result']].drop_duplicates(),
                       right_on=[id_col],
                       left_on=[id_col], how='left')

    df_intermittency = df.copy()
    df_intermittency['intermittency'] = df_intermittency.groupby([id_col])[name_qty_col].transform(
        compute_intermitency)
    
    results = pd.merge(results, df_intermittency[[id_col, 'intermittency']].drop_duplicates(),
                       right_on=[id_col],
                       left_on=[id_col], how='left')
    
    #'------------------Intermittency/ADI analysis computed------------------'#
    #
    # compute relevant flags using acf and pacf
    #df_relevant_lags_acf = df.groupby([id_col])[name_qty_col].apply(
    #    lambda x: find_acf_relevant_lags(x)).reset_index().rename(columns={name_qty_col: 'acf_lags'})
    #
    #logger.info('-----------------AIC analysis computed-----------------')
    #results = pd.merge(results, df_relevant_lags_acf,
    #                  right_on=[id_col],
    #                  left_on=[id_col], how='left')
    #
    # df_relevant_lags_pacf = df.groupby([id_col])[name_qty_col].apply(
    #    lambda x: find_pacf_relevant_lags(x)).reset_index().rename(columns={name_qty_col: 'pacf_lags'})
    #
    # print('-----------------PAIC analysis computed-------- \n')
    # results = pd.merge(results, df_relevant_lags_pacf,
    #                   right_on=[id_col],
    #                   left_on=[id_col], how='left')

    return results

