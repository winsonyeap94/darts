# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:53:03 2020

@author: alvaro.garcia.piquer
"""

import pandas as pd
import xlsxwriter
import os
from xlrd import XLRDError

import data_acquisition as da

INPUT_PATH = "input" + os.path.sep
OUTPUT_PATH = "output" + os.path.sep

def data_structure_validation(template_df, table_columns, primary_keys):
       
    #Columns init
    template_df['Missing'] = True
    #template_df['All_nan'] = True
    template_df['Primary_key'] = False
    
    #Mark the columns that exists in the data as False
    template_df.loc[template_df['Field'].str.lower().isin(table_columns), 'Missing'] = False
    
    #Mark the columns that contains at least one value different to NaN as False
    '''#Change to sql query
    nan_columns = data.isna().all()
    nan_columns = nan_columns[nan_columns]
    template_df.loc[~template_df['Field'].isin(list(nan_columns.index)), 'All_nan'] = False'''
    
    #Mark the columns that are primary key as True
    template_df.loc[template_df['Field'].isin(primary_keys), 'Primary_key'] = True
    
    return template_df

def validate_all_tables(db_engine, database, s3_template_bucket, template_file, s3_data_buckets):
    #SQL info retrieved from s3 buckets that contain tables, it is necessary to identify the primary keys
    folders_info = {}
    for s3_data_bucket in s3_data_buckets:
        folders_info.update(da.obtain_all_files('s3', s3_data_bucket))
    
    #A list with the tables in the data request file are obtained
    template_tables = da.read_excel_file(s3_template_bucket, template_file,
                                     "Overview", skiprows=[0])
    all_tables_in_db = db_engine.execute_query("SHOW TABLES IN " + database)
    all_tables_in_db = da.serie_to_str_list(all_tables_in_db, 'tab_name')

    validated_tables = {}
    for table_name in template_tables['Table']:
        #Retrieve the information of the table in the data request
        try:
            template_df = da.read_excel_file(s3_template_bucket, template_file,
                                     table_name, skiprows=[0])
            template_df = template_df.rename(columns=lambda x: x.strip())
        except XLRDError:
            validated_tables[table_name] = "Missing Excel Sheet"
            continue
        #Retrieve the table from the database
        if table_name.lower() in all_tables_in_db:
            #Obtain the columns of the table
            table_columns = db_engine.execute_query("SHOW COLUMNS IN " + database + "." + table_name)
            table_columns = da.serie_to_str_list(table_columns, 'field')
            #Structure of the table is validated
            template_df = data_structure_validation(template_df, table_columns, folders_info[table_name]['primary_keys'])
            pk_repetitions = primary_key_validation_sql(db_engine, database, table_name, folders_info[table_name]['primary_keys'])
            template_df['Primary_key_repeated'] = False
            if len(pk_repetitions) > 0:
                template_df.loc[template_df['Field'].isin(folders_info[table_name]['primary_keys']), "Primary_key_repeated"] = True
            #Validation of the table is stored
            #template_df.to_csv('./output/' + table_name+'.csv', index=False, sep=';')
            validated_tables[table_name] = template_df
        else:
            validated_tables[table_name] = "Missing Database Table"
    return validated_tables

def primary_key_validation_df(data, primary_keys):
    #Check if primary keys are unique
    pk_summary = {}
    #Stores the primary keys columns
    pk_summary['columns'] = primary_keys
    groupped = data.groupby(primary_keys)
    #Stores the number of unique values of the primary keys
    pk_summary['different_values'] = len(groupped) 
    #Stores the repeated values of the primary keys
    repetitions = groupped.size() > 1
    pk_summary['repetitions'] = repetitions[repetitions]
    return pk_summary

def primary_key_validation_sql(db_engine, database, table_name, primary_keys):
    #res = da.execute_query(conn, "select " + ','.join(primary_keys) + ',count(*) as repetitions from ' + database + "." + table_name + " group by " + ','.join(primary_keys))
    
    res = db_engine.execute_query("select * from (select " + ','.join(primary_keys) + ',count(*) as repetitions from ' + database + "." + table_name + " group by " + ','.join(primary_keys) + ") where repetitions > 1")

    return res

def continent_validation(continent_name, data, field, continent_file):
    data_countries = data.groupby(field)
    continent_countries = pd.read_csv(continent_file, sep=";")
    #We obtain the data countries in the continent countries
    country_summary = {}
    countries_match = list(set(data_countries.groups.keys()) & set(continent_countries['code']))
    countries_match_names = continent_countries[continent_countries['code'].isin(countries_match)]['country']
    country_dict = pd.Series(continent_countries['code'].values,index=continent_countries['country']).to_dict()
    country_summary[continent_name] = {}
    for c in countries_match_names:
        c_code = country_dict[c]
        country_summary[continent_name][c] = data_countries.size()[c_code]
    return country_summary

def save_validation(table_dict, output_file):
    writer = pd.ExcelWriter(output_file)
    for k, v in table_dict.items():
        #Write table in a new sheet
        if isinstance(v, str):
            worksheet = writer.book.add_worksheet(k)
            worksheet.write(0, 0, str(v))
        else:
            v_filt = v[(v['Missing']==True) | (v['Primary_key_repeated']==True)]
            v_filt = v_filt[v_filt['Request'].isin(['Mandatory', 'Optional'])]
            v_filt.to_excel(writer, index=False, sheet_name=k)
    writer.save()
    writer.close()
    
def save_validation_summary(column_summary, pk_summary):
    #The name of the output file is the original data file with "_validation" suffix
    output_file = OUTPUT_PATH + "data_validation" + ".xlsx"
    writer = pd.ExcelWriter(output_file)
    #Write column summary in a new sheet
    column_summary.to_excel(writer, index=False, sheet_name='Columns')
    
    #Write summaries in text in a new sheet
    worksheet = writer.book.add_worksheet('PrimaryKeys')
    row = 0
    for col, values in pk_summary.items():
        worksheet.write(0, row, col)    
        worksheet.write(1, row, str(values))
        row += 1
    writer.save()
    writer.close()

def add_table_header(table_file, query_file):
    #We obtain from the query file the information of the table
    #table_name is a list containing the database and the name of the table
    #columns is a list containing all the columns existing in the table
    #constraints is a list containing the constraints defined (it is not used)
    #primary_keys is a list containing the primary keys of the table
    with open(query_file, 'r') as file: 
        line = file.readline()
        table_name, columns, constraints, primary_keys = obtain_headers(line)
    
    #Read table and add the columns read
    data = pd.read_csv(table_file, names=columns)   
    #The name of the output file is the table name
    output_file = OUTPUT_PATH + table_name[1] + ".csv"
    data.to_csv(output_file, index=False)
    return data, table_name, constraints, primary_keys
    
def validations(data, template_file, table_name, constraints, primary_keys):
    #Next, we validate the obtained information with the data request template
    #column_summary contains the table of the template file with three new columns (Missing, All_nan, PrimaryKey)
    column_summary = data_structure_validation(data=data,
                        template_file = template_file,
                        table_name = table_name,
                        constraints=constraints,
                        primary_keys=primary_keys)
    #pk_summary is a dictionary containing some primary keys analysis
    pk_summary = primary_key_validation(data, primary_keys)
    
    country_summary = continent_validation(continent_name="Europe", 
                                           data=data, 
                                           field="LAND1", 
                                           continent_file=INPUT_PATH + 'europe_codes.csv')
    
    #A new excel file is created with the status of each column and the primary keys analysis
    save_validation(column_summary, pk_summary)
    return column_summary, pk_summary, country_summary
         

if __name__ == "__main__":
    data, table_name, constraints, primary_keys = add_table_header(table_file = INPUT_PATH + 'data.csv', 
                                                           query_file = INPUT_PATH + 'create.sql')
    
    column_summary, pk_summary, country_summary = validations(data=data,
                                                            template_file = INPUT_PATH + 'D.A.R.E._IO OPS Inventory_Visibility_DataRequest.xlsx',
                                                            table_name = table_name,
                                                            constraints=constraints,
                                                            primary_keys=primary_keys)
    