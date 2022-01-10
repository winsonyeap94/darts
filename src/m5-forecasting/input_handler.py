"""
Input handler for managing any data loading. 

This script may also contain some minor preprocessing. For more complex data preprocessing, it may be worth to perform
it in a separate script.
"""

import pandas as pd

from base.decorators import log_time


class InputHandler:

    @classmethod
    @log_time
    def get_processed_data(cls, summarise_monthly=True):
        """
        Main data getter function, includes data preprocessing:
        - Melting sales data into long table format
        - Adding calendar information to sales
        """

        sales_df, calendar_df, sell_prices_df = cls.get_all_data()

        # Melting calendar dataset into a long table format
        sales_df = sales_df[['item_id', 'store_id', 'item_store_id'] + [x for x in sales_df.columns if x.startswith('d_')]]
        long_sales_df = pd.melt(sales_df, id_vars=['item_store_id', 'item_id', 'store_id'], 
                                var_name='d', value_name='sales')

        # Merging with calendar_df to get date information
        long_sales_df = long_sales_df.merge(calendar_df[['date', 'd']], on='d', how='left')
        long_sales_df['date'] = pd.to_datetime(long_sales_df['date'])

        # Summarising data (if enabled)
        final_sales_df = long_sales_df.copy()[['item_id', 'store_id', 'date', 'sales']]
        if summarise_monthly:
            final_sales_df['yearmonth'] = final_sales_df['date'].dt.to_period('M')
            final_sales_df = final_sales_df.groupby(['item_id', 'store_id', 'yearmonth']).sum().reset_index(drop=False)

        # Converting 'yearmonth' datatype from Period to regular 'datetime'
        final_sales_df['yearmonth'] = final_sales_df['yearmonth'].apply(lambda x: x.to_timestamp())

        return final_sales_df

    @classmethod
    def get_all_data(cls):
        """
        Secondary data getter function. Consolidates data from all sources and returns them.
        """

        # Calendar data
        calendar_df = cls.get_calendar_data()

        # Sell Prices data
        sell_prices_df = cls.get_sell_prices_data()

        # Sales data
        sales_df = cls.get_sales_data()
        
        return sales_df, calendar_df, sell_prices_df

    @staticmethod
    def get_calendar_data():
        calendar_df = pd.read_csv("../../data/m5-forecasting-accuracy/calendar.csv")
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])
        return calendar_df

    @staticmethod
    def get_sell_prices_data():
        sell_prices_df = pd.read_csv("../../data/m5-forecasting-accuracy/sell_prices.csv")
        return sell_prices_df

    @staticmethod
    def get_sales_data():

        # Raw Data
        sales_df = pd.read_csv("../../data/m5-forecasting-accuracy/sales_train_validation.csv")

        # Creating a primary key fr "item-store" level
        sales_df['item_store_id'] = sales_df['item_id'].astype(str) + '-' + sales_df['store_id'].astype(str)
        
        return sales_df


if __name__ == "__main__":

    data_df = InputHandler.get_processed_data()
    