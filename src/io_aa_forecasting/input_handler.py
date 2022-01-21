"""
Input handler for managing input data loading. 

This script may also contain some minor preprocessing. For more complex data preprocessing, it may be worth to perform
it in a separate script.
"""

import os
import pandas as pd
from pandarallel import pandarallel

from base import loguru_logger
from base.decorators import log_time
import TS_forecast_functions as ts_fcst


class InputHandler:
    """
    Main class for loading the input data required for model training.

    Note:
    1. External factors are loaded separately in the external_factors_handler.py script.
    """

    def __init__(self, forecast_level='DFU', target_var='demand_volume'):

        self.forecast_level = forecast_level
        self.target_var = target_var
        self._logger = loguru_logger

    @log_time
    def get_processed_data(self, plant_scope, start_date, last_date, backtest_periods, lag, load_cached_data=False):
        """
        Main data getter function, includes data preprocessing:
        """

        demand = self.get_apo_demand_data(plant_scope, start_date, last_date)
        master_data = self.get_apo_product_master_data()
        segmentation = self.get_forecast_segmentation_data()

        # Selecting only DFUs which are found in apo_product_md (active_qualification == 'X' only)
        master_data = master_data.loc[master_data['DFU'].isin(master_data['DFU'].unique())]

        # Define snapshots
        snapshots = ts_fcst.define_snapshots(df=demand, backtest_periods=backtest_periods, lag=lag)

        # Prepare data in the right format
        # Mapping from DFU to the desired forecast level
        if self.forecast_level != "DFU":
            if ((type(self.forecast_level) == list) & (len(self.forecast_level) > 1)):
                master_data["newDFU"] = master_data[self.forecast_level].astype(str).agg('_'.join, axis=1)
                self.forecast_level = "newDFU"
                df_dfu = master_data[["DFU", self.forecast_level]].drop_duplicates()
            elif ((type(self.forecast_level) == list) & (len(self.forecast_level) == 1)):
                forecast_level = self.forecast_level[0]
                df_dfu = master_data[["DFU", self.forecast_level]].drop_duplicates()
            else:
                df_dfu = master_data[["DFU", self.forecast_level]].drop_duplicates()
        else:
            df_dfu = master_data[["DFU"]].drop_duplicates()
            
        # Build demand to be used (filter products in scope, delete obsolete and select target variable)
        demand, demand_dfu, _, _, _, _ = ts_fcst.prepare_data(
            demand=demand, df_seg=segmentation, df_dfu=df_dfu, 
            forecast_level=self.forecast_level, target_var=self.target_var
        )

        # Print DFUs in scope
        if self.forecast_level == "DFU":
            demand["plant_code"] = demand["DFU"].apply(lambda x: x[-4:])
            self._logger.info(demand.groupby(["plant_code"])["DFU"].nunique().reset_index(name="#DFU in scope"))
            demand = demand.drop("plant_code", axis=1)
        else:
            self._logger.info(str(demand[self.forecast_level].nunique()) + " #" + self.forecast_level + " in scope")

        return demand, master_data, segmentation, snapshots

    @log_time
    def get_apo_demand_data(self, plant_scope, start_date='', last_date=''):
        
        demand = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/dare_apo_demand.csv"))
        demand = demand.rename(columns={'dfu':'DFU'})

        # Filtering based on start_date and last_date
        if start_date != "":
            demand = demand.loc[demand['demand_period'] >= start_date]
        if last_date != "":
            demand = demand.loc[demand["demand_period"] < last_date]

        # Filter demand for the scope if it is not an empty list
        if len(plant_scope) > 0:
            demand["plant_code"] = demand["DFU"].apply(lambda x: x[-4:])
            demand = demand.loc[demand["plant_code"].isin(plant_scope)]
            demand = demand.drop("plant_code", axis=1)

        max_date = demand['demand_period'].max()

        # pandarallel.initialize()
        # demand = demand.groupby(['DFU']).parallel_apply(lambda x: self.extend_dates(x, self.target_var, max_date))\
        #     .reset_index()
        demand = demand.groupby(['DFU']).apply(lambda x: self.extend_dates(x, self.target_var, max_date))\
            .reset_index()
        demand = demand[['DFU', 'demand_period', self.target_var, 'demand_period_new']]

        return demand

    @staticmethod
    @log_time
    def get_apo_product_master_data():

        apo_product_md = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/dare_apo_product_md.csv"))

        # Quick fix for growing data
        apo_product_md = apo_product_md.drop_duplicates(['material_code','plant_code'], keep = 'last')

        # Renaming columns
        apo_product_md = apo_product_md.rename(columns={'dfu':'DFU',
                                                        'active_apo':'active_APO',
                                                        'active_mrp':'active_MRP'})

        # Take X only
        apo_product_md = apo_product_md.loc[apo_product_md['active_qualification'].isin(['x', 'X'])].drop_duplicates()
        
        return apo_product_md

    @staticmethod
    @log_time
    def get_forecast_segmentation_data():
        
        segmentation_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/forecast_segmentation.csv"))
        segmentation_df = segmentation_df.rename(columns={'dfu':'DFU',
                                                          'abc':'ABC',
                                                          'xyz':'XYZ',
                                                          'adi':'ADI',
                                                          'adi_result':'ADI_result',
                                                          'cv_result':'CV_result'})
        
        return segmentation_df

    @staticmethod
    def extend_dates(df: pd.DataFrame, target_var: str, end_date: str):
        """Extend demand_period for DFU not up to last_date"""

        df = df[['demand_period', target_var]]
        
        if pd.to_datetime(df['demand_period'].max()) < pd.to_datetime(end_date):
            # Create extension
            date_extension = pd.date_range(start=pd.to_datetime(df['demand_period'].max()), end=pd.to_datetime(end_date), freq='MS')
            df_extension = pd.DataFrame(columns=df.columns)
            df_extension['demand_period'] = date_extension.strftime('%Y-%m')
            df_extension[target_var] = 0
        
            df = pd.concat([df, df_extension], axis=0)
            
        df['demand_period_new'] = pd.to_datetime(df['demand_period'])
        
        return df


if __name__ == "__main__":

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

    demand, master_data, segmentation, snapshots = InputHandler().get_processed_data(
        plant_scope=plant_scope, start_date=start_date, last_date=last_date, backtest_periods=backtest_periods, lag=lag
    )
    print(f"demand: \n{demand.head()}\n")
    print(f"master_data: \n{master_data.head()}\n")
    print(f"segmentation: \n{segmentation.head()}\n")
    print(f"snapshots: \n{snapshots}\n")
    