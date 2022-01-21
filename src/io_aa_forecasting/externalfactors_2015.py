# -*- coding: utf-8 -*-
"""
Created on 29-March-2021

@author: jan.heinrich.hayer
"""
# General modules 

import os
import pandas as pd 
import numpy as np
from datetime import datetime, date, timedelta
import json 
import sys 
import requests

# Exchange Rates 
from forex_python.converter import CurrencyRates

# Modules for working calendar
import calendar
from workalendar.europe.united_kingdom import UnitedKingdom
from workalendar.europe.denmark import Denmark
from workalendar.america.mexico import Mexico
from workalendar.europe.russia import Russia
from workalendar.usa.north_carolina import NorthCarolina
from workalendar.europe.germany import Germany
from workalendar.europe.spain import Spain
from workalendar.europe.czech_republic import CzechRepublic
from workalendar.europe.turkey import Turkey
from workalendar.europe.france import France
from workalendar.asia.japan import Japan
from workalendar.asia.singapore import Singapore

# Modules for miles driven 
from fredapi import Fred

sys.path.append(os.path.join(os.path.dirname(__file__), '../l1l3/aux_functions/'))
import db_snowflake_management

# #Database engine
# db_mode = 'redshift'

# if db_mode == 'athena':
#     db_engine = db_athena_management.DBAthena()
#     read_database ='dare_iops'
#     write_database ='dare_iops'
#     rejection_condition = ""
# else:
#     db_engine = db_snowflake_management.Snowflake()
#     read_database ='dare_l1'
#     write_database ='dare_pre_hm'
#     rejection_condition = "sys_rejected is null"
# db_engine.connect()



# Working calendar 

class calendarDays(): 
    '''
    Returns calendar days 
    '''
    def __init__(self, first_year, last_year): 
        self.country_mappings = {'2700': 'AE', '0120': ['united_kingdom', UnitedKingdom()], '2230': ['denmark', Denmark()], 
                                 '0400': ['mexico', Mexico()], '2400': ['russia', Russia()],'0559': ['north_carolina', NorthCarolina()], '0041': ['germany', Germany()], 
                                 '0600': ['spain', Spain()], '0700': ['czech_republic', CzechRepublic()], 
                                 '0750': 'BA', '2600': ['turkey', Turkey()], '0950': ['france', France()], '3540': ['japan', Japan()], '2900': ['singapore', Singapore()]}
            
        self.first_year = first_year
        self.last_year = last_year 
            
            
    def gdayofmonth(self): 
        '''
        :returns: Gets one day per month. 
        '''
        years = list(range(self.first_year, self.last_year))
        months = list(range(1, 13))
        output = []
        for year in years: 
            for month in months: 
                start = datetime(year, month, 1)
                lastday = calendar.monthrange(year, month)[1]
                end = datetime(year, month, lastday)
                output.append((start, end))
        return output 
    
    
    def getframe(self):
        '''
        :returns: Dataframe with calendar.
        '''
        dd = self.gdayofmonth()
        df_calendar = pd.DataFrame(dd)
        
        # On the fly definition 
        f = lambda x: cal.get_working_days_delta(x[0], x[1])

        for key, value in self.country_mappings.items(): 
            if isinstance(value, list): 
                cal = value[1]
                df_calendar[key] = df_calendar.apply(f, axis=1)

        df_calendar['month'] = pd.to_datetime(df_calendar[0]).apply(lambda x: x.strftime("%Y-%m"))
        df_calendar.drop(columns=[0, 1], inplace=True)
        return df_calendar
    
    
# Miles Driven     

class milesDriven(): 
    '''
    Wrapper for https://github.com/mortada/fredapi. 
    Currently returning series -> https://fred.stlouisfed.org/series/TRFVOLUSM227SFWA. 
    Can be adjusted by the output of fred.search('miles'). 
    '''
    def __init__(self, api_key): 
        self.ak = api_key # Apply from https://research.stlouisfed.org/docs/api/api_key.html
        self.series = 'TRFVOLUSM227NFWA' # DEFAULT TO 'M12MTVUSM227NFWA'
        

    def get_milesdriven(self):
        '''
        Returns time series "Vehicle Miles Traveled" in million of miles and seasonally adjusted. 
        '''
        fred = Fred(api_key=self.ak)
        print('Getting series M12MTVUSM227NFWA')
        return fred.get_series('M12MTVUSM227NFWA')
        
        

    
   # Exchange Rate 
class execQuery: 
    '''
    Class for executing query from .txt file. 
    '''
    def __init__(self, plant_str):
        self.query_output = dict()
        self.counter = 0
        self.colnames = ['location_from_id', 'location_to_id', 'movement_qty_agg', 'country_to', 'currency_to', 'country_from', 'currency_from' ]
        self.query = """WITH cte_table AS(
                                    SELECT location_from_id, location_to_id, movement_qty_agg, country_to, currency_to, country_from, currency_from
                                    FROM 
                                    (SELECT location_from_id, location_to_id, SUM(movement_qty) AS movement_qty_agg
                                    FROM dare_l3.dare_network
                                    WHERE  len(location_to_id) = 4 AND location_from_id IN ({plant})
                                    GROUP BY location_from_id, location_to_id) AS A LEFT JOIN (SELECT plant_id, currency AS currency_to, country_to  
                                              FROM dare_temp.country_currency) AS B  ON A.location_to_id = B.plant_id
                                                LEFT JOIN (SELECT plant_id, currency AS currency_from, country_to AS country_from 
                                                    FROM dare_temp.country_currency) AS C  ON A.location_from_id = C.plant_id
UNION 
                                SELECT location_from_id, location_to_id, movement_qty_agg, A.country_to, currency_to, country_from,  currency_from
                                FROM 
                                (SELECT location_from_id, location_to_id, movement_qty_agg, country_to, country_from, currency_from
                                 FROM 
                                (SELECT location_from_id, location_to_id, SUM(movement_qty) AS movement_qty_agg
                                FROM dare_l3.dare_network
                                WHERE  len(location_to_id) <> 4 AND location_from_id IN ({plant})
                                GROUP BY location_from_id, location_to_id) AS A LEFT JOIN (SELECT DISTINCT(customer), country AS country_to 
                                           FROM dare_l1.bw_cust_master bcm) AS D ON D.customer = A.location_to_id
                                                LEFT JOIN (SELECT plant_id, currency AS currency_from, country_to AS country_from 
                            FROM dare_temp.country_currency) AS C  ON A.location_from_id = C.plant_id) AS A LEFT JOIN (SELECT DISTINCT(country_to) AS country_to, currency AS currency_to  
                            FROM dare_temp.country_currency) AS B ON A.country_to = B.country_to
                    ) SELECT * 
                      FROM cte_table 
                      WHERE currency_to IS NOT NULL AND currency_to <> currency_from AND movement_qty_agg > 0""".format(plant=plant_str)
       
    @staticmethod
    def json_load(f):
        '''
        Transforms the database output to a json. 
        :returns: Extracted data as json format.  
        '''
        return [json.loads(k) for k in f]
     
    def create_df(self): 
        '''
        Creates a dataframe based on the query_output dictionary. 
        '''
        generator = (pd.DataFrame(self.json_load(value)) for value in self.query_output.values())
        df = pd.concat(generator)
        df.columns = self.colnames 
        return df
    
    def creat_df_concat(self):
        df = pd.concat(self.query_output.values())
        df.columns = [col.lower() for col in df.columns]
        
        return df
    
    def execute_query(self):
        '''
        Executes the query as appends output to instance attribute query_output w/ key being the counter.
        '''
        #Database engine
        db_mode = 'redshift'

        if db_mode == 'athena':
            db_engine = db_athena_management.DBAthena()
            read_database ='dare_iops'
            write_database ='dare_iops'
            rejection_condition = ""
        else:
            db_engine = db_snowflake_management.Snowflake()
            read_database ='dare_l1'
            write_database ='dare_pre_hm'
            rejection_condition = "sys_rejected is null"
        db_engine.connect()
        
        #self.query_output[self.counter] = db_engine.execute_query(self.query, rds_query_lambda_name=rds_query_lambda_name, verbose=False)
        self.query_output[self.counter] = db_engine.execute_query(self.query, rds_query_lambda_name=None, verbose=False)
        self.counter += 1
        
    def create_weighting(self): 
        '''
        Calculates weights on a plant level. 
        :params df: weights dataframe 
        :returns: Returns dataframe. 
        '''
        #df = self.create_df()
        df = self.creat_df_concat()
        df = df.groupby(['currency_from', 'location_from_id', 'currency_to'])['movement_qty_agg'].sum().reset_index()
        df['weighting'] = df.groupby(['currency_from'])['movement_qty_agg'].apply(lambda x: x/(np.sum(x)))
        return df
    
    
    
class twIndex(): 
    
    def __init__(self): 
        pass 
    
    def daypermonth(self):
        """
        Returns one day per month.
        :params year: Year from which extraction start. 
        """
        output = list()
        for year in list(range(2015, date.today().year + 1)):
            for month in list(range(1, 13)): 
                if date(year, month, 15) <= date.today(): 
                    output.append(date(year, month, 15))
        return output 
    
    def getcurrency(self, currencies): 
        '''
        Returns currency for the dates and currency specified. 
        This method is using free version which now has very limited usage (https://exchangeratesapi.io/pricing/). Use getcurrent_api method instead. 
        :params currencies: Currencies to be returned. As list. 
        :params dates: Dates as list. 
        :returns: Dict with currency as key. 
        '''
        dates = self.daypermonth() # now as list after fix. 
        # Inst class 
        df_cr = dict()
        for currency in currencies: 
            c = CurrencyRates()
            try: 
                c = {d.strftime('%Y-%m-%d'): c.get_rates(currency, d) for d in dates}
                df_cr[currency] = pd.DataFrame(c).T   
            except:
                pass
        df_cr = {k:1/v for (k,v) in df_cr.items()}
        return df_cr
    
    @staticmethod
    def get_exchangerateapi_response(date, base_curr):
        '''Return API response for rates based on a specified date and base currency 
        The API key used for the api call is on monthly paid subscription plan (https://exchangeratesapi.io/pricing/). 
        '''
        url = 'http://api.exchangeratesapi.io/v1/'
        paid_api_key = 'fa8ca388f5ee3d1ed708be3927b62d75'
        
        response_rates = None
        try:
            response = requests.get(url + '{}?access_key={}&base={}&format=1'.format(date, paid_api_key, base_curr)).json()
            response_rates = response["rates"]
        except:
            pass
        return response_rates
    
    
    def getcurrency_api(self, currencies):
        '''
        Returns currency for the dates and currency specified with API call. 
        :params currencies: Currencies to be returned. As list. 
        :params dates: Dates as list. 
        :returns: Dict with currency as key. 
        '''
        dates = self.daypermonth()
        df_cr = dict()
        for currency in currencies:
            try:
                c = {d.strftime('%Y-%m-%d') : self.get_exchangerateapi_response(d, currency) for d in dates}
                df_cr[currency] = pd.DataFrame(c).T 
            except:
                pass
        df_cr = {k: v for k, v in df_cr.items() if v is not None} # remove key if value is none from dictionary
        df_cr = {k:1/v for (k,v) in df_cr.items()}
        return df_cr
            
    
    def tw_index(self, weights): 
        '''
        :params weights: Weights dataframe. 
        '''
        
        df_cr = self.getcurrency_api(weights['currency_from'].unique().tolist())
        
        tw_ci_d = dict()
        for loc in weights['location_from_id'].unique().tolist(): 
            df_weights = weights.loc[weights['location_from_id'] == loc]

            key = df_weights['currency_from'].unique()[0]
     
            df_crs = df_cr.get(key)
            if df_crs is not None: 
        
                cols = df_crs.columns.intersection(df_weights['currency_to'].tolist())
                df_crs = df_crs[cols]
                df_weights = df_weights.loc[df_weights['currency_to'].isin(cols)]
                
                df_crs = df_crs / df_crs.iloc[0]
                
                sorterIndex = dict(zip(df_crs.columns.tolist(), range(len(df_crs.columns.tolist()))))
                df_weights['rank'] = df_weights['currency_to'].map(sorterIndex)
                df_weights.sort_values('rank', inplace=True)
                weights_list = df_weights[~df_weights['rank'].isna()]['weighting'].tolist()
                
                tw_ci = np.sum(df_crs.mul(weights_list, axis=1), axis=1)
                tw_ci = tw_ci.to_frame(name='tw_ci')
                
                tw_ci = tw_ci.reset_index().rename(columns={'index': 'date'})
                tw_ci['month'] = pd.to_datetime(tw_ci['date']).apply(lambda x: x.strftime("%Y-%m"))
                
                tw_ci_d[df_weights['location_from_id'].unique()[0]] = tw_ci
                
        return tw_ci_d
    
    

    
    
    
# Sales Prediction 
class salesPrediction(): 
    
    def __init__(self): 
        #self.path = '/home/ec2-user/SageMaker/r-d/data_analysis/output/spdata_31032021.csv'
        self.path = os.path.join(os.path.dirname(__file__), 'output/spdata_31032021.csv')
        
    def load(self): 
        '''
        :returns: Dataframe sales prediction. 
        '''
        df_spred = pd.read_csv(self.path)
        df_spred.rename(columns={'\nPDB: Sales Prediction for Sales Designation': 'sales_pred'}, inplace=True)
        assert df_spred.shape[0] != 0, 'No data loaded'
        return df_spred
    
    def get_material(self, mat_list): 
        '''
        :params mat_list: List with materials. 
        :returns: Material with names. 
        '''
        
        # #Database engine
        # db_mode = 'redshift'

        # if db_mode == 'athena':
        #     db_engine = db_athena_management.DBAthena()
        #     read_database ='dare_iops'
        #     write_database ='dare_iops'
        #     rejection_condition = ""
        # else:
        #     db_engine = db_snowflake_management.Snowflake()
        #     read_database ='dare_l1'
        #     write_database ='dare_pre_hm'
        #     rejection_condition = "sys_rejected is null"
        # db_engine.connect()
        
        # mat_str = "'"+"','".join(mat_list)+"'"
        # #query = "active_qualification = 'X' AND active_mrp = 'X0' AND active_apo IN (1,2,3, '1', '2', '3') AND material_name IN ({})".format(mat_str)
        # query = "active_qualification = 'X' AND active_mrp = 'X0' AND active_apo IN (CAST(1 AS VARCHAR),CAST(2 AS VARCHAR),CAST(3 AS VARCHAR), '1', '2', '3') AND material_name IN ({})".format(mat_str)
        # df = db_engine.execute_select_query('dare_pre_hm', "dare_apo_product_md", "material_name,DFU", condition=query)

        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/mat_df.csv"))
        
        df.drop_duplicates(inplace=True)
        
        return df 
    
    def create_index(self): 
        '''
        :returns: Cleaned sales prediction dataframe. 
        '''
        df_spred = self.load()
        df_spred = df_spred.groupby(['material_name', 'year'])['sales_pred'].sum().reset_index()
        
        # Get DFU name 
        mat_list = df_spred['material_name'].unique().tolist()
        df_mats = self.get_material(mat_list)
        df_spred = pd.merge(df_spred, 
                            df_mats, 
                            how='inner', 
                            on='material_name')
        
        # Filter out zeros. Break down to monthly level 
        df_spred = df_spred.loc[df_spred['sales_pred'] > 0]
        df_spred['sales_pred'] = np.where(df_spred['year'].isin(['2018', '2019', '2020']), df_spred['sales_pred']/12, df_spred['sales_pred']/12) 
        df_spred.sort_values(['material_name', 'year'], inplace=True)
        df_spred.drop_duplicates(['material_name', 'year'], inplace=True)
        return df_spred
    


    
    
    
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# # Vehicles In Operation 
# class vehiclesOperation():
    
#     def __init__(self): 
#         self.path = '/home/ec2-user/SageMaker/r-d/data_analysis/output/20210315_VIO data.xlsx'
        
#     def load(self): 
#         sheets = [str(i) for i in list(range(2018, 2022))]
#         tables_vio = {i: pd.read_excel(self.path, sheet_name=i, engine='openpyxl') for i in sheets}
#         assert len(tables_vio) != 0, "No data loaded"
#         return tables_vio
        
#     def cleaning(self): 
#         tables_vio = self.load()
#         output_vio = {}
#         for key in tables_vio: 
#             df = tables_vio[key].rename(columns={'PDB: Product Name': 'material_name','Vehicle Age': 'age', 
#                                                  '\nPDB: Vehicle Population': 'population', 
#                                                  '\n\nPDB: Replacement Rate': 'replacement'})
#             df['Sales Designation'] = df['Sales Designation'].fillna(method='ffill')
#             df['material_name'] = df['material_name'].fillna(method='ffill')
#             df['year'] = key
#             df.drop(df[df['age'] == 'Result'].index, inplace=True)
#             # In 2019 *9
#             df['age'] = df['age'].replace('*9', '9') # Might be deleted! 
#             df['age'] = df['age'].astype(float)
#             df = df.loc[df['age'] <= 14]
#             output_vio[key] = df
#         df_vio = pd.concat(output_vio)  
#         df_vio.reset_index(drop=True, inplace=True)
#         return df_vio 
        
#     def create_index(self): 
#         df_vio = self.cleaning()
        
#         # Sum population  
#         df_vio_agg = df_vio.groupby(['material_name', 'age', 'year'])['population'].sum().reset_index()
        
#         # Calculate index 
#         df_vio_agg['ap'] = df_vio_agg['age'] * df_vio_agg['population'] 
#         aux = df_vio_agg.groupby(['material_name', 'year'])[['ap', 'population']].sum().reset_index()
#         aux['index'] = (aux['ap'] / aux['population']) / 14
#         df_vio_agg = pd.merge(df_vio_agg, aux, on=['material_name', 'year'], how='left')
#         df_vio_agg.drop(columns=['ap_x', 'ap_y', 'population_y', 'population_x', 'age'], inplace=True)
#         df_vio_agg.drop_duplicates(['material_name', 'year'], inplace=True)
#         return df_vio_agg
    
#     def get_material(self, mat_list): 
#         mat_str = "'"+"','".join(mat_list)+"'"
#         query = "active_qualification = 'X' AND active_mrp = 'X0' AND active_apo IN (1,2,3, '1', '2', '3') AND material_name IN ({})".format(mat_str)
#         df = db_engine.execute_select_query('dare_pre_hm', "dare_apo_product_md", "material_name,DFU", condition=query)
#         df.drop_duplicates(inplace=True)
#         return df 
        
#     def return_vio(self): 
#         df_vio_agg = self.create_index()
#         mats_list = df_vio_agg['material_name'].unique().tolist()
#         df_mats = self.get_material(mats_list)
#         # Merge 
#         df_vio_agg = pd.merge(df_vio_agg, 
#                               df_mats, 
#                               how='inner', 
#                               on='material_name')
#         df_vio_agg.sort_values(['material_name', 'year'], inplace=True)
#         df_vio_agg.drop_duplicates(['material_name', 'year'], inplace=True)
#         return df_vio_agg  
    
    
    
    
    
    
    
    
    
    
    
    
# # Exchange Rate 
# class execQuery: 
#     '''
#     Class for executing query from .txt file. 
#     '''
#     def __init__(self, plant_str):
#         self.query_output = dict()
#         self.counter = 0
#         self.colnames = ['location_from_id', 'location_to_id', 'movement_qty_agg', 'country_to', 'country_from', 'currency_from', 'currency_to']
#         self.query = """SELECT location_from_id, location_to_id, movement_qty_agg, country_to, country_from, currency_from, currency_to
#                                 FROM 
#                                 (SELECT location_from_id, location_to_id, SUM(movement_qty) AS movement_qty_agg
#                                 FROM dare_l3.dare_network
#                                 WHERE location_from_id IN ({plant}) AND len(location_to_id) = 4
#                                 GROUP BY location_from_id, location_to_id) AS A LEFT JOIN (SELECT plant_id, country_name AS country_to
#                                                                                             FROM dare_pre_hm.dare_location_md dlm) AS B  ON A.location_to_id = B.plant_id
#                                                                                 LEFT JOIN (SELECT plant_id, country_name AS country_from
#                                                                                             FROM dare_pre_hm.dare_location_md dlm) AS C  ON A.location_from_id = C.plant_id
#                                                                                 LEFT JOIN (SELECT currency AS currency_from, plant_id 
#                                                                                            FROM dare_l1.bw_mm_purchasing bmp 
#                                                                                            WHERE sys_date = '2021-03-03'
#                                                                                            GROUP BY currency, plant_id) AS D ON A.location_from_id = D.plant_id
#                                                                                 LEFT JOIN (SELECT currency AS currency_to, plant_id 
#                                                                                            FROM dare_l1.bw_mm_purchasing bmp 
#                                                                                            WHERE sys_date = '2021-03-03'
#                                                                                            GROUP BY currency, plant_id) AS E ON A.location_to_id = E.plant_id
#                                                                                 WHERE country_to <> country_from """.format(plant=plant_str)
       
#     @staticmethod
#     def json_load(f):
#         '''
#         Transforms the database output to a json. 
#         :returns: Extracted data as json format.  
#         '''
#         return [json.loads(k) for k in f]
     
#     def create_df(self): 
#         '''
#         Creates a dataframe based on the query_output dictionary. 
#         '''
#         generator = (pd.DataFrame(self.json_load(value)) for value in self.query_output.values())
#         df = pd.concat(generator)
#         df.columns = self.colnames 
#         return df
    
#     def execute_query(self):
#         '''
#         Executes the query as appends output to instance attribute query_output w/ key being the counter.
#         '''
#         self.query_output[self.counter] = db_engine.execute_query(self.query, rds_query_lambda_name=rds_query_lambda_name, verbose=False)
#         self.counter += 1
        
#     def create_weighting(self): 
#         '''
#         Calculates weights on a plant level. 
#         :params df: weights dataframe 
#         :returns: Returns dataframe. 
#         '''
#         df = self.create_df()
#         df = df.groupby(['currency_from', 'location_from_id', 'currency_to'])['movement_qty_agg'].sum().reset_index()
#         df['weighting'] = df.groupby(['currency_from'])['movement_qty_agg'].apply(lambda x: x/(np.sum(x)))
#         return df
    