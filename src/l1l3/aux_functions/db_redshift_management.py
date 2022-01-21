# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:28:56 2020

@author: alvaro.garcia.piquer
"""
import db_management
import data_acquisition as da
from sqlalchemy import create_engine
from sqlalchemy import text
import psycopg2
import boto3
import json
import pandas as pd
import ast
import math
from joblib import Parallel, delayed
import time
from datetime import datetime
from tqdm import tqdm
import os

REDSHIFT_PARAMETERS = {'resource_name': 's3', 'athena_bucket_name': 'dare-iops-athena-data', 'region_name': 'eu-central-1'}


class DBRedshift(db_management.DB):
    
    def __init__(self):
        self.ssm = boto3.client('ssm')
        self.region = 'eu-central-1'  # todo: somehow cannot read from env variables. so making manual for now
        self.iam_role = self.ssm.get_parameter(Name='/iam_role/id').get("Parameter").get("Value")
    
    def connect(self):
        #jdbc:redshift://iopsredshift-redshiftcluster-1jqtbtvrthx6f.cenzdcifqpa9.eu-central-1.redshift.amazonaws.com:5439/iopsredshiftdb
        #cnxn = psycopg2.connect('DRIVER={Amazon Redshift (x64)}; Server=iopsredshift-redshiftcluster-1jqtbtvrthx6f.cenzdcifqpa9.eu-central-1.redshift.amazonaws.com; Database=iopsredshiftdb; UID=reduser; Port=5439')
        #con=psycopg2.connect(dbname= 'dbname', host='host', port= 'port', user= 'user', password= 'pwd')'
        pass
        
    def execute_query(self, query, rds_query_lambda_name, verbose=True, response=True, executions=0, retries=2):
        executions += 1
        try:
            payload = {"RequestType": "Query",
                       "Query": query
                      }
            client = boto3.client('lambda')


            if verbose:
                print(query)
            invoke_response = client.invoke(FunctionName=rds_query_lambda_name,
                                           InvocationType='RequestResponse', #'Event'
                                           Payload=json.dumps(payload)
                                            )
            if verbose:
                print("Query executed")
            resp = invoke_response['Payload'].read().decode('utf-8')
            resp_json = json.loads(resp)
            try:
                data = json.loads(resp_json['Data'])
            except:
                try:
                    data = resp_json['Data']
                except:
                    data = resp_json
                if response:
                    raise ValueError("Error:", data, "in query\n\t",query)        
        except:
            display("exception, trial " + str(executions))
            if executions > retries:
                raise ValueError("Error:", data, "in query\n\t",query)
            else:
                self.execute_query(query, rds_query_lambda_name, verbose, response, executions, retries)
                
        return data
        
    def execute_select_query(self, database, table, fields, condition='', block_size=10000, verbose=False, threads=24, direct_conection=False):
        
        start_time = time.time()
        
        # select environment and adequate lambda name
        environment = da.getenv()
        rds_query_lambda_name = da.get_redshift_query_lambda_name(environment)

        if isinstance(fields, list):
            fields = ",".join(fields)
        if block_size is None or threads is None:
            query = "select " + fields + " from " + database + "." + table
            if condition.strip() != '':
                query += " where " + condition
            data_list = self.execute_query(query, verbose=verbose, rds_query_lambda_name=rds_query_lambda_name)
        else:
            # condition added so only min and max sys_id is selected for rows matching the condition
            if condition.strip() != '':
                sys_condition = " where " + condition
            else:
                sys_condition = ""

            count_query = "select count(sys_id), min(sys_id), max(sys_id) from " + database + "." + table + sys_condition
            count_res = self.execute_query(count_query, verbose=verbose, rds_query_lambda_name=rds_query_lambda_name)[0].replace("[","").replace("]","")
            count = int(count_res.split(",")[0])
            if count == 0:
                empty_df = pd.DataFrame()
                return empty_df
            minr = int(count_res.split(",")[1])-1 #substract one to get also the first row one in the first sys_id < chunk
            maxr = int(count_res.split(",")[2])  
            
            iterations = math.ceil((maxr - minr) / block_size)
            if verbose:
                display(str(count) + " rows found from " + str(minr) + " to " + str(maxr) + " ("+str(iterations)+" iterations)")

            query = "select " + fields + " from " + database + "." + table + " where "
            if condition.strip() != '':
                query += condition + " and "
                
            data = []    
            parallel=True
            if parallel:
                data = Parallel(n_jobs=threads, prefer="threads")(delayed(self.execute_query)(query = query + "sys_id >  " + str(minr + i * block_size) + " and sys_id <= " + str(minr + i*block_size + block_size), verbose=verbose, rds_query_lambda_name=rds_query_lambda_name) for i in tqdm(range(0, iterations)))
                
            else:
                print("not parallel")
                for i in range(0, iterations):
                    print(i)
                    data.append(self.execute_query(query + "sys_id >  " + str(minr + i * block_size) + " and sys_id <= " + str(minr + i*block_size + block_size), verbose=verbose, rds_query_lambda_name=rds_query_lambda_name))
                    print(query +  "sys_id >  " + str(i * block_size) + " and sys_id <= " + str(i*block_size + block_size))

            data_list = [item for sublist in data for item in sublist]
            print(len(data_list))
            
            print("Query Completed: %s seconds" % (time.time() - start_time))
        
        if fields.strip() == "*":
            columns_query = "select column_name from INFORMATION_SCHEMA.COLUMNS where TABLE_NAME = '" + table + "' and table_schema = '" + database + "' ORDER BY ORDINAL_POSITION"
            columns = self.execute_query(columns_query, verbose=verbose, rds_query_lambda_name=rds_query_lambda_name)
            columns = [value.replace('[', '') for value in columns]
            columns = [value.replace(']', '') for value in columns]
            columns = [value.replace('"', '') for value in columns]
        else:
            columns = fields.split(",")
        
        start_time = time.time()
        #for i in range(len(data_list)):
            #data_list[i] = data_list[i].replace('true', '"t"')
            #data_list[i] = data_list[i].replace('null', '"null"').replace('true', '"t"')
        
        #data_list = Parallel(n_jobs=-1)(delayed(data_list[i].replace)('null', '""') for i in range(0, len(data_list)))
        
        #if "".join(data_list[0:5]) == "ERROR":
        #    raise ValueError('Error in the following query:', query)
        print("Frame Cleansed: %s seconds" % (time.time() - start_time))
        start_time = time.time()
        data_list = [json.loads(k) for k in data_list]
        data_list = pd.DataFrame(data_list, columns = columns) 
        print("Frame Created: %s seconds" % (time.time() - start_time))
        
        # check if query has the right number of rows:
        #if count != n_rows:
        #    raise ValueError('Error in query. Number of rows in source table: ', count, ' do not match with obtained rows: ', n_rows)
        
        return data_list
    
    def execute_join_query(self, database, table1, table2, fields1, fields2, on1, on2, drop_duplicates=True, join='inner', verbose=False, block_size=10000, threads=70):
        # select environment and adequate lambda name
        environment = da.getenv()
        rds_query_lambda_name = da.get_redshift_query_lambda_name(environment)
        
        temp_database = "dare_temp"
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        temp_table = "temp_join_" + table1 + "_" + table2 + "_" + timestamp
        query = self.create_join_query(database, table1, table2, fields1, fields2, on1, on2, drop_duplicates, join)
        query = "create table " + temp_database + "." + temp_table + " as (" + query + ")"
        if verbose:
            display(query)
        #Execute join and store it in a temporal table
        res = self.execute_query(query, rds_query_lambda_name=rds_query_lambda_name, verbose=verbose, response=False)
        
        #Select from temporal table
        df = self.execute_select_query(temp_database, temp_table, "*", condition='', block_size=block_size, verbose=verbose, threads=threads)
        
        #Drop temporal table
        drop_query = "drop table " + temp_database + "." + temp_table
        drop_res = self.execute_query(drop_query, rds_query_lambda_name=rds_query_lambda_name, verbose=verbose, response=False)
        return df
    
    def execute_write_query(self, query):
        payload = {"RequestType": "Query",
                   "Query": query
                  }
        client = boto3.client('lambda')
        
        #'iopsredshift-DBQueryLambdaFn-KC3QOLOBAA3O'
        #'iopsdbfrankfurt-DBQueryLambdaFn-1X4GPBMDS2NBW'
        
        # select environment and adequate lambda name
        environment = da.getenv()
        rds_query_lambda_name = da.get_redshift_query_lambda_name(environment)

        invoke_response = client.invoke(FunctionName=rds_query_lambda_name,
                                       InvocationType='RequestResponse',
                                       Payload=json.dumps(payload)
                                    )
        #resp = invoke_response['Payload'].read().decode('utf-8')
        resp = ""
        return resp
    
    def execute_write_df(self, df, schema, table, chunksize=50000):
        if chunksize > len(df):
            chunksize = len(df)
        list_df = [df[i:i+chunksize] for i in range(0,df.shape[0],chunksize)]
        print("Chunks found", len(list_df))
        #c = 0
        for df_chunk in list_df:
            #print("Chunk",c,"starting with",df_chunk.iloc[0].material_id,"and length",len(df_chunk))
            engine = create_engine('sqlite://', echo = False) 
            df_chunk.to_sql('tmp', con = engine, index=False)  
            query = engine.execute("SELECT * FROM tmp").fetchall()
            self.execute_write_query("INSERT INTO " + schema + "." + table + " VALUES " + ",".join(str(e) for e in query)) 
            #c += 1
            #print("INSERT INTO " + schema + "." + table + " VALUES " + ",".join(str(e) for e in query))
       
    def copy_chunk(self, chunk_id, df_chunk, s3_bucket, schema, table, verbose=False):
        table_name = schema + "." + table
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        s3_file_name, _ = da.save_df_to_db(df_chunk, s3_bucket, None, None, table+"_temp_"+str(chunk_id)+"_"+timestamp, sep=";")
        if verbose:
            display("s3://" + s3_bucket + "/" + s3_file_name)
        
        # query = "copy " + table_name + " from 's3://" + s3_bucket + "/" + s3_file_name + "' iam_role 'arn:aws:iam::944917888538:role/iopsredshift-RedshiftS3AccessRole-1NFGIZ5Z3N23K' delimiter ';' region 'eu-central-1' ignoreheader 1 removequotes emptyasnull blanksasnull;"
        query = f"copy {table_name} from 's3://{s3_bucket}/{s3_file_name}' iam_role '{self.iam_role}' delimiter ';' region '{self.region}' ignoreheader 1 removequotes emptyasnull blanksasnull; "
        # query = f"copy {table_name} from 's3://{s3_bucket}/{s3_file_name}' delimiter ';' region '{self.region}' ignoreheader 1 removequotes emptyasnull blanksasnull; "

        #quote as '\"'
        res = self.execute_write_query(query)
        if verbose:
            display(query)
        s3 = boto3.resource('s3')
        s3.Object(s3_bucket, s3_file_name).delete()
        return res
    
    def create_table_from_df(self, s3_bucket, df, schema, table, drop_table=True, chunksize=30000, verbose=False, parallel=True, force_drop_table=False, delete_table = False, delta = 60):
        table_name = schema + "." + table
        
        if drop_table:
            # empties (truncte) the current table while keeping its structure
            self.execute_write_query("truncate table " + table_name + ";")
        if force_drop_table:
            # completely remove (drop) the table
            self.execute_write_query("drop table " + table_name + ";")
        if delete_table: 
            # deletes newest records of the destination table so new ones can be uploaded
            today = datetime.today().strftime('%Y-%m-%d')
            day_limit = str(pd.to_datetime(today) - pd.to_timedelta(delta, unit='d'))[0:10]
            day_limit = int(day_limit.replace('-', ''))
            self.execute_write_query("delete from " + table_name + " where day > " + str(day_limit) + ";")
        
        #Create the table if it does not exist
        query = pd.io.sql.get_schema(df.reset_index(), 'table_name')
        query = query.replace('"table_name"', table_name)
        query = query.replace('\n"index" INTEGER,','')
        query = query.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS")
        self.execute_write_query(query)
        
        if chunksize is not None:
            if chunksize > len(df):
                chunksize = len(df)
            list_df = [df[i:i+chunksize] for i in range(0,df.shape[0],chunksize)]
        else:
            list_df = [df]
    
        if verbose:    
            print("Chunks found", len(list_df))
        
        if not parallel:
            for df_chunk in list_df:
                #Create file in s3
                s3_file_name, _ = da.save_df_to_db(df_chunk, s3_bucket, None, None, table+"_temp", sep=";")
                display("s3://" + s3_bucket + "/" + s3_file_name)
                # query = "copy " + table_name + " from 's3://" + s3_bucket + "/" + s3_file_name + "' iam_role 'arn:aws:iam::944917888538:role/iopsredshift-RedshiftS3AccessRole-1NFGIZ5Z3N23K' delimiter ';' region 'eu-central-1' IGNOREHEADER 1 removequotes emptyasnull blanksasnull;"
                query = f"copy {table_name} from 's3://{s3_bucket}/{s3_file_name}' iam_role '{self.iam_role}' delimiter ';' region '{self.region}' IGNOREHEADER 1 removequotes emptyasnull blanksasnull; "
                # query = f"copy {table_name} from 's3://{s3_bucket}/{s3_file_name}' delimiter ';' region '{self.region}' IGNOREHEADER 1 removequotes emptyasnull blanksasnull; "

                #quote as '\"'
                self.execute_write_query(query)
                display(query)

                s3 = boto3.resource('s3')
                s3.Object(s3_bucket, s3_file_name).delete()
        else:
            # parallel execution:
            # 1 - create folder
            timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
            s3_folder_name = table+"_temp_"+timestamp
            temporal_files = []
            # 2 - create a file for each chunk
            for i in range(0, len(list_df)):
                df_chunk = list_df[i]
                #Create file in s3
                s3_file_name, _ = da.save_df_to_db(df_chunk, s3_bucket, None, None, s3_folder_name+"_"+str(i), sep=";", folder_name=s3_folder_name)
                temporal_files.append(s3_file_name)
                if verbose:
                    display("s3://" + s3_bucket + "/" + s3_file_name)
            # 3 - copy of the entire folder
            # query = "copy " + table_name + " from 's3://" + s3_bucket + "/outputs/" + s3_folder_name + "' iam_role 'arn:aws:iam::944917888538:role/iopsredshift-RedshiftS3AccessRole-1NFGIZ5Z3N23K' delimiter ';' region 'eu-central-1' IGNOREHEADER 1 removequotes emptyasnull blanksasnull;"
            query = f"copy {table_name} from 's3://{s3_bucket}/outputs/{s3_folder_name}' iam_role '{self.iam_role}' delimiter ';' region '{self.region}' IGNOREHEADER 1 removequotes emptyasnull blanksasnull; "
            # query = f"copy {table_name} from 's3://{s3_bucket}/outputs/{s3_folder_name}' delimiter ';' region '{self.region}' IGNOREHEADER 1 removequotes emptyasnull blanksasnull; "

            #quote as '\"'
            self.execute_write_query(query)
            if verbose:
                display(query)

            # delete all temporal files
            s3 = boto3.resource('s3')
            for f in temporal_files:
                s3.Object(s3_bucket, f).delete()
        
        '''else:
            #We assume that df only contains NaN in str columns
            #TODO: convert to copy chunks (lambda)
            self.execute_write_df(df.fillna('') , schema, table, chunksize)'''
