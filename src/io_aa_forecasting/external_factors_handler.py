"""
Input handler for loading external factors.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

from base import loguru_logger
from base.decorators import log_time
from externalfactors_2015 import calendarDays, milesDriven, salesPrediction, execQuery, twIndex


class ExternalFactorsHandler:
    """
    Similar to InputHandler, but responsible for external factors.
    1. Exchange Rate
    2. Sales Orders
    3. Working Days
    4. Miles Driven
    5. Sales Prediction
    """

    def __init__(self, demand, plant_scope, execution_date, start_date, apply_sales_orders):

        self.max_demand_period = demand['demand_period'].max()
        self.plant_scope = plant_scope
        self.execution_date = execution_date
        self.start_date = start_date
        self.apply_sales_orders = apply_sales_orders
        
        self._logger = loguru_logger

    @log_time
    def load_external_factors(self):
        """Main function for loading all external factors."""

        self.df_tw_ci = self.load_exchange_rate(self.plant_scope)
        self.df_orders_g = self.load_sales_order(self.max_demand_period)
        self.df_wd = self.load_working_days(self.execution_date)
        self.df_md, self.df_md_future = self.load_miles_driven(self.start_date)
        self.df_spred = self.load_sales_prediction()

    @log_time
    def load_exchange_rate(self, plant_scope):
        
        self._logger.debug(f"{plant_scope=}")
        plant_str = "'" + "','".join(plant_scope) + "'"

        # cw = execQuery(plant_str)
        # cw.execute_query()
        # df = cw.creat_df_concat()
        # df_weighting = cw.create_weighting()

        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/cw_df.csv"))
        df_weighting = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/cw_df_weighting.csv"))
        
        df_weighting.drop(df_weighting[df_weighting['weighting'] == 1].index, inplace=True)
        tw_ci = twIndex().tw_index(df_weighting)

        # Create dataframe
        df_tw_ci = pd.DataFrame()
        for plant_code in tw_ci.keys():
            temp_df = tw_ci[plant_code]
            temp_df['plant_code'] = plant_code
            temp_df['P'] = temp_df['month'].apply(lambda x: x.split('-')[1])
            
            df_tw_ci = pd.concat([df_tw_ci, temp_df], axis=0)
            
        df_tw_ci['date_new'] = pd.to_datetime(df_tw_ci['month'])

        return df_tw_ci

    @log_time
    def load_sales_order(self, max_demand_period):

        # Load orders
        df_orders = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/external_orders_load_duplicate.csv"), 
                                dtype='str')
        df_orders['sys_date'] = pd.to_datetime(df_orders['sys_date'])
        df_orders[['conf_qty', 'corr_qty', 'order_qty']] = \
            df_orders[['conf_qty', 'corr_qty', 'order_qty']].apply(lambda x: pd.to_numeric(x, errors='coerce'))
        
        # Filter for next month 
        # next_month_fcst =(df_orders['sys_date'].max() + relativedelta(months=+1)).strftime("%Y-%m")

        # next_month_fcst = (pd.to_datetime(demand.demand_period.max()) + relativedelta(months=+2)).strftime("%Y-%m")
        next_month_fcst = (pd.to_datetime(max_demand_period) + relativedelta(months=+2)).strftime("%Y-%m")
        df_orders = df_orders.loc[df_orders['req_month'] == next_month_fcst].reset_index(drop=True)

        if df_orders.shape[0] != 0:
            df_orders.loc[(df_orders['conf_qty'] != df_orders['corr_qty']), 'conf_qty'] = df_orders['corr_qty']
            df_orders = df_orders.loc[df_orders['conf_qty'] >= 0]
            df_orders = df_orders.loc[df_orders['conf_qty'] != 0]
            
            df_orders_g = df_orders.groupby(['material', 'req_month', 'sys_date', 'plant'])['conf_qty']\
                .sum()\
                .reset_index()\
                .sort_values('req_month')
            df_orders_g['DFU'] = df_orders_g['material'] + "_" + df_orders_g['plant']
            df_orders_g = df_orders_g.rename(columns={"req_month": "period", 
                                                      "sys_date": "forecast_date", 
                                                      "conf_qty": "so_quantity"})
            df_orders_g['forecast_date'] = pd.to_datetime(df_orders_g['forecast_date'])\
                .apply(lambda x: x.strftime("%Y-%m"))

        else: 
            self.apply_sales_orders = False
            self._logger.info('No orders loaded')

        if self.apply_sales_orders:
            self._logger.debug(f"{df_orders_g.period.min()=}, {df_orders_g.period.max()=}")

        return df_orders_g

    @log_time
    def load_working_days(self, execution_date):
        
        wd_max_year = (datetime.strptime(execution_date, "%Y-%m") + relativedelta(years=2)).year
        self._logger.debug(f"{wd_max_year=}")
        df_calendar = calendarDays(2015, wd_max_year).getframe()
        df_wd = df_calendar.melt(id_vars = ['month'], var_name = 'plant_code', value_name = 'working_days')
        df_wd['date_new'] = pd.to_datetime(df_wd['month'])
        self._logger.debug(f"{df_wd['plant_code'].unique()=}")
        self._logger.debug(df_wd.head(2))
        self._logger.debug(df_wd.tail(5))
        self._logger.debug(f"{df_wd.date_new.min()=}, {df_wd.date_new.max()=}")

        return df_wd

    @log_time
    def load_miles_driven(self, start_date):

        # api_key = 'fa5527959d0b0c2cd14196e7780fffe8'

        # md = milesDriven(api_key).get_milesdriven()
        # df_md = md.to_frame().reset_index()

        df_md = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/miles_driven.csv"))
        df_md.rename(columns={df_md.columns[0]: 'date_new', df_md.columns[1]: 'miles_driven'}, inplace=True)
        df_md['month'] = pd.to_datetime(df_md['date_new']).apply(lambda x: x.strftime("%Y-%m"))
        df_md['plant_code'] = '0559' # US plant
        self._logger.debug(f"{df_md.month.min()=}, {df_md.month.max()=}")

        # Limit start date
        if start_date != "":
            df_md = df_md.loc[df_md['month'] >= start_date]
            
        # Initialize to None. This will be updated with arima result at every snapshot.
        df_md_future = None  # TODO: The original script uses this as a global variable, need to think of how to avoid that.

        return df_md, df_md_future

    @log_time
    def load_sales_prediction(self):

        df_spred = salesPrediction().create_index()
        assert df_spred['year'].dtypes == np.int64, 'Transform year!'
        df_spred = df_spred.rename(columns={'dfu':'DFU'})

        return df_spred

    @staticmethod
    def _add_Y_P_transform(df, group_by_var):
        
        if group_by_var == 'plant_code':
            df['plant_code'] = df['DFU'].apply(lambda x: x[-4:])
        
        df['Y'] = df["demand_period"].apply(lambda x: x.split('-')[0])
        df['P'] = df["demand_period"].apply(lambda x: x.split('-')[1])
        df["Y_vol"] = df.groupby([group_by_var])['demand_volume'].transform("sum")
        df["P_vol"] = df.groupby([group_by_var, "P"])['demand_volume'].transform("sum")
        
        return df[[group_by_var, "Y", "P", "Y_vol", "P_vol"]]

    def get_disag_weights(self, df_demand, snapshot_date):
        
        # Get weights to disaggregate tw_ci to dfu level
        demand_agg = df_demand.copy() ## demand for plant_code
        demand_disag = df_demand.copy() ## demand for dfu
        
        demand_agg['demand_period_new'] = pd.to_datetime(demand_agg['demand_period'])
        demand_disag['demand_period_new'] = pd.to_datetime(demand_disag['demand_period'])

        demand_agg = demand_agg[demand_agg.demand_period_new < snapshot_date]
        demand_disag = demand_disag[demand_disag.demand_period_new < snapshot_date]

        demand_agg = self._add_Y_P_transform(demand_agg, 'plant_code')
        demand_disag = self._add_Y_P_transform(demand_disag, 'DFU')

        # Merge by plant_code and P
        demand_disag['plant_code'] = demand_disag['DFU'].apply(lambda x: x[-4:])
        weights = pd.merge(demand_disag.drop_duplicates(), demand_agg.drop_duplicates(), on=['plant_code', "Y", "P"], how="left").dropna()

        # Calculate yearly and period weights
        weights["weight_Y"] = (weights["Y_vol_x"] / weights["Y_vol_y"])
        weights["weight_P"] = (weights["P_vol_x"] / weights["P_vol_y"])

        # # Normalize weights
        min_max_weights = weights.groupby(['plant_code']).agg({'weight_Y':['min', 'max'],
                                                                'weight_P':['min', 'max']}).reset_index()
        min_max_weights.columns= ['plant_code', 'min_weight_Y', 'max_weight_Y', 'min_weight_P', 'max_weight_P']
        weights = pd.merge(weights, min_max_weights, on=['plant_code'], how='left')
        weights['weight_Y'] = (weights['weight_Y'] - weights['min_weight_Y'] + 1e-6) / \
            (weights['max_weight_Y'] - weights['min_weight_Y'] + 1e-6)
        weights['weight_P'] = (weights['weight_P'] - weights['min_weight_P'] + 1e-6) / \
            (weights['max_weight_P'] - weights['min_weight_P'] + 1e-6)

        weights = weights[['DFU',"Y", "P", "weight_Y", "weight_P"]].drop_duplicates()

        # Obtain final weight and normalize it
        weights["weight"] = weights[["weight_Y", "weight_P"]].mean(axis=1)

        # Select relevant columns
        weights = weights[["DFU", "P", "weight"]]

        # Correct nan weights
        weights = weights.pivot_table("weight", ["DFU"], "P").reset_index()
        weights = weights.melt(id_vars=["DFU"], var_name="P", value_name='weight')
        weights.loc[weights.weight.isna(), "weight"] = 0
        weights = weights.sort_values(["DFU", "P"])
        
        return weights

    @staticmethod
    def apply_SO_correction(df_forecast, df_orders_g):
        """
        Enriches the forecast with the sales orders. 

        Args:
            df_forecast (pd.DataFrame): Forecast dataframe. 
            df_orders_g (pd.DataFrame): Orders dataframe. 

        Returns:
            pd.DataFrame: Enriched forecast.
        """
        
        df_forecast_so = pd.merge(df_forecast.rename(columns={'forecast_period': 'period'}), 
                                  df_orders_g, 
                                  on=["forecast_date", "DFU", "period"],
                                  how = "left")
        df_forecast_so["final_quantity"] = df_forecast_so["quantity"]
        df_forecast_so.loc[
            (df_forecast_so["so_quantity"] > df_forecast_so["quantity"]) & ~(df_forecast_so["so_quantity"].isna()), 
            "final_quantity"
        ] = df_forecast_so["so_quantity"]
        df_forecast_so = df_forecast_so.drop(columns=['quantity'])
        df_forecast_so = df_forecast_so.rename(columns={"period":"forecast_period", 
                                                        "final_quantity":"quantity"})
        df_forecast_so = df_forecast_so[['DFU', 'forecast_period', 'quantity', 'forecast_date', 'forecast_lag', 
                                         'type', 'model']]
        
        return df_forecast_so

    def get_future_wd(self, dfu: str, df_wd: pd.DataFrame, future_min_date: pd.datetime, horizon=12):
        """
        Get future dataframe from preloaded workingdays dataframe with date range determined by future_min_date and 
        horizon.
        """
        
        plant_code = dfu[-4:] 
        future_max_date = future_min_date + relativedelta(months=horizon + 2) # date range for future dataframe
        print (f"future_min_date: {future_min_date} | future_max_date: {future_max_date}")

        future_wd = df_wd.loc[(df_wd['plant_code']==plant_code) & (df_wd["date_new"] > future_min_date) & (df_wd["date_new"] <= future_max_date)]
        future_wd = future_wd[['date_new', 'working_days']].rename(columns={'date_new':'ds'}).reset_index(drop=True)
        future_wd = future_wd.sort_values(['ds'])
        
        return future_wd

    def get_future_md(self, dfu: str, df_md_future: pd.DataFrame, future_min_date: pd.datetime, horizon=12):
        """
        Get future dataframe from preloaded milesdriven dataframe with date range determined by future_min_date and 
        horizon.
        """

        plant_code = dfu[-4:] 
        future_max_date = future_min_date + relativedelta(months=horizon + 2) # date range
        self._logger.debug(f"future_min_date MD: {future_min_date} | future_max_date MD: {future_max_date}")

        future_md = df_md_future.loc[
            (df_md_future['plant_code']==plant_code) & (df_md_future["date_new"] > future_min_date) & \
            (df_md_future["date_new"] <= future_max_date), :
        ].copy()
        future_md = future_md[['date_new', 'miles_driven']].rename(columns={'date_new':'ds'}).reset_index(drop=True)
        future_md = future_md.sort_values(['ds'])

        return future_md

    @staticmethod
    def perp_demand(demand, df_spred, c=0.9):
        """
        Calculates correlation between demand and sales prediction and return a list of DFUs with correlation greater than c. 

        Args:
            demand (pd.DataFrame): Demand DataFrame.
            df_spred (pd.DataFrame): Sales Prediction DataFrame. 
            c (float, optional): Correlation coefficient. Defaults to 0.9.
        """

        # Aggregates demand to a yearly level and then breaks it down again to monthly level. Was done as not all years are complete. 
        # Sales prediction already at monthly level as the yearly figure was divided by 12. 
        demand_aux = demand.copy()
        demand_aux['year'] = pd.to_datetime(demand_aux['demand_period'], format="%Y-%m").dt.year
        
        # Get parameters
        divisor_first_year = 12 - pd.to_datetime(demand_aux['demand_period'].min()).month + 1
        years = np.sort(demand_aux['year'].unique())
        divisor_last_year = pd.to_datetime(demand_aux['demand_period'].max()).month 
        
        demand_agg = demand_aux.groupby(['DFU', 'year'])['demand_volume'].sum().reset_index()
        
        demand_agg.loc[demand_agg['year'] == years[0], 'demand_volume'] = demand_agg['demand_volume'] / divisor_first_year
        demand_agg.loc[demand_agg['year'] == years[-1], 'demand_volume'] = demand_agg['demand_volume'] / divisor_last_year
        demand_agg.loc[demand_agg['year'].isin(years[1:-1]), 'demand_volume'] = demand_agg['demand_volume'] / 12
        
        # Join demand and sales prediction  
        aux = pd.merge(df_spred, 
                       demand_agg[demand_agg['year'].isin(years[:-1])], 
                       on=['DFU', 'year'], 
                       how='inner')
        
        # Correlation 
        corr = aux.groupby('DFU')[['demand_volume', 'sales_pred']].corr().iloc[0::2, -1]\
            .reset_index()\
            .rename(columns={'sales_pred': 'correlation'})
        dfu_list = corr[corr['correlation'] >= c]['DFU'].tolist()
        
        return dfu_list

    @staticmethod
    def get_change(df_spred, demand, dfu_list, year=2021):
        """
        Calculates the percentage change for sales prediction data and applies this trend to the demand to get a demand 
        forecast for the year specified in the func input. 

        Args:
            demand (pd.DataFrame): Demand DataFrame.
            df_spred (pd.DataFrame): Sales Prediction DataFrame. 
            dfu_list ([type]): [description]
            year (int, optional): Demand year to be forecasted. Defaults to 2021.

        Returns:
            pd.DataFrame: New demand table with the forecasted demand for the year specified (year).
        """
        
        # Creates a list with year and year-1 -> [2020, 2021]  
        years = list()
        years.append(year)
        years.insert(0, year-1)
        
        # Keep DFUs in sales prediction with high correlation and calculate percentage change of the year [2020, 2021]  
        df_spred = df_spred.loc[df_spred['DFU'].isin(dfu_list)]
        df_spred = df_spred.loc[df_spred['year'].isin(years)]
        df_spred['pct_change_sales'] = df_spred.groupby('DFU')['sales_pred'].apply(pd.Series.pct_change)
        df_spred.dropna(inplace=True)
        
        # Merge aggregated demand and the sales prediction. 
        demand_aux = demand.copy()
        demand_aux['year'] = pd.to_datetime(demand_aux['demand_period'], format="%Y-%m").dt.year
        demand_agg = demand_aux.loc[demand_aux['year'] == years[0]].groupby(['DFU', 'year'])['demand_volume'].mean().reset_index()
        demand_prediction = pd.merge(demand_agg, 
                                     df_spred, 
                                     on=['DFU'], 
                                     how='inner')
        
        # *12 to bring to yearly level avagain. 
        demand_prediction['new_demand'] = \
            (demand_prediction['demand_volume'] * demand_prediction['pct_change_sales']) + \
            demand_prediction['demand_volume'] * 12
            
        return demand_prediction

    @staticmethod
    def apply_SP_correction(demand_new, forecast, year=2021): 
        """
        Applies the new demand to enrich the forecast. 

        Args:
            demand_new (pd.DataFrame): Demand dataframe (output from get_change).
            forecast (pd.DataFrame): Forecast DataFrame.
            year (int, optional): Demand year to be forecasted. Defaults to 2021.
        """
        
        forecast['year_period'] = pd.to_datetime(forecast['forecast_period'], format="%Y-%m").dt.year
        # Merge forecast with demand
        forecast_c = pd.merge(forecast[(forecast['year_period'] == year)], 
                              demand_new[['DFU', 'new_demand']], 
                              on=['DFU'], 
                              how='inner')
        
        # Apply enrichement. 
        forecast_c['new_demand_m'] = forecast_c['new_demand'] / 12
    #     forecast_c['final_quantity'] = (forecast_c['new_demand_m'] - forecast_c['quantity']) / forecast_c['new_demand_m'] * forecast_c['quantity'] + forecast_c['quantity']
        forecast_c['final_quantity'] = (forecast_c['new_demand_m'] - forecast_c['quantity']) / 2 + forecast_c['quantity']
        forecast_c.loc[forecast_c['final_quantity'] < 0, 'final_quantity'] = forecast_c['quantity']
        
        # Merge with all forecast data. 
        forecast_merged = pd.merge(forecast_c[['DFU', 'forecast_date', 'forecast_period', 'final_quantity']],
                                   forecast, 
                                   right_on=['DFU', 'forecast_date', 'forecast_period'],
                                   left_on=['DFU', 'forecast_date', 'forecast_period'], 
                                   how='right')
        
        forecast_merged.loc[forecast_merged['final_quantity'].isna(), 'final_quantity'] = forecast_merged['quantity']
        forecast_merged.rename(columns={'forecast_period': 'period'}, inplace=True)
        
        return forecast_merged


if __name__ == "__main__":

    # Forecast configurations
    plant_scope = ["0041", "2400", "2230", "2600", "0120", "0559", "0600", "0700", "0750", "0950"]
    start_date = "2015-01"
    execution_date = "2019-11"
    last_date = execution_date
    periods = 0
    lag = 1
    missingMonth = 1

    if periods > 0:
        backtest_periods = periods + 4 - 1 + missingMonth
    elif periods == 0:
        backtest_periods = periods

    # Initialised as True, assuming External Factors are enabled
    apply_sales_orders = True

    # Loading demand data from InputHandler
    from input_handler import InputHandler
    demand, master_data, segmentation, snapshots = InputHandler().get_processed_data(
        plant_scope=plant_scope, start_date=start_date, last_date=last_date, backtest_periods=backtest_periods, lag=lag
    )

    # External Factors Handler
    ext_factors_handler = ExternalFactorsHandler(
        demand=demand, plant_scope=plant_scope, execution_date=execution_date, start_date=start_date,
        apply_sales_orders=apply_sales_orders
    )
    ext_factors_handler.load_external_factors()

    df_tw_ci = ext_factors_handler.df_tw_ci
    df_orders_g = ext_factors_handler.df_orders_g
    df_wd = ext_factors_handler.df_wd
    df_md = ext_factors_handler.df_md
    df_spred = ext_factors_handler.df_spred

    print(f"df_tw_ci: \n{df_tw_ci.head()}\n")
    print(f"df_orders_g: \n{df_orders_g.head()}\n")
    print(f"df_wd: \n{df_wd.head()}\n")
    print(f"df_md: \n{df_md.head()}\n")
    print(f"df_spred: \n{df_spred.head()}\n")
