# -*- coding: utf-8 -*-
"""
Created on 02-Sep-2020

@author: alvaro.garcia.piquer; manuel.blanco.fraga
"""

import boto3
import pandas as pd
    
def read_excel_file(s3_bucket, file, tab, skiprows=0):
    """
    Function to read an excel file from a S3 bucket
    """
    data_location = 's3://{}/{}'.format(s3_bucket, file)
    df = pd.read_excel(data_location, tab, skiprows=skiprows)
    return df

def read_csv_file(s3_bucket, file, skiprows=0, sep=";", decimal=".", header = "infer"):
    """
    Function to read a csv file from a S3 bucket
    """
    data_location = 's3://{}/{}'.format(s3_bucket, file)
    df = pd.read_csv(data_location, skiprows=skiprows, sep=sep, decimal=decimal, header = header)
    return df

def write_csv_file(s3_bucket, file, dataframe: pd.DataFrame, sep=";", decimal=".", index=False, mode = "w", header=True):
    """
    Function to write a dataframe to a csv file from a S3 bucket
    """
    data_location = 's3://{}/{}'.format(s3_bucket, file)
    dataframe.to_csv(data_location, sep=sep, decimal=decimal, index=index, mode=mode, header=header)
    return None

