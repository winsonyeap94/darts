from snowflake import connector
from snowflake.connector.pandas_tools import write_pandas
import db_management
import pandas as pd
import time
from datetime import datetime
import boto3


class Snowflake(db_management.DB):

    def __init__(self):
        super().__init__()
        ssm = boto3.client("ssm")
        self.username = ssm.get_parameter(Name='/snowflake/user').get('Parameter').get('Value')
        self.password = ssm.get_parameter(Name='/snowflake/pwd', WithDecryption=True).get('Parameter').get('Value')
        self.sf_account = ssm.get_parameter(Name='/snowflake/account').get('Parameter').get('Value')
        self.region = ssm.get_parameter(Name='/snowflake/region').get('Parameter').get('Value')
        self.sf_wh = ssm.get_parameter(Name='/snowflake/warehouse').get('Parameter').get('Value')
        self.db = ssm.get_parameter(Name='/snowflake/database').get('Parameter').get('Value')
        self.current_schema = 'dare_l1'

        self.ctx = None
        self.cs = None
        if self.ctx is None:
            self.connect()
            self.cs = self.ctx.cursor()

    def connect(self):
        # Connecting to Snowflake using the default authenticator
        self.ctx = connector.connect(
            user=self.username,
            password=self.password,
            account=self.sf_account,
            warehouse=self.sf_wh,
            database=self.db,
            schema=self.current_schema
        )
        print(self.ctx)

    def query(self, query):
        print('execute query:', query)

        try:
            self.cs.execute(query)
            df = self.cs.fetch_pandas_all()
        except Exception as e:
            print("snowflake query error", e)

        return df

    def execute_query(self, query, verbose=True, response=True, executions=0, retries=2, rds_query_lambda_name=None):
        return self.query(query)

    def execute_select_query(self, database, table, fields, condition='', verbose=False, threads=24,
                             direct_conection=False):
        start_time = time.time()

        if isinstance(fields, list):
            fields = ",".join(fields)

        if condition.strip() != '':
            sys_condition = " where " + condition
        else:
            sys_condition = ""

        count_query = "select count(sys_id), min(sys_id), max(sys_id) from " + self.db + '.' + database + "." + table + sys_condition
        count_res = self.query(count_query)
        print("count_res result", "\n", count_res)

        minr = count_res['MIN(SYS_ID)'].values[0] - 1
        maxr = count_res['MAX(SYS_ID)'].values[0]

        query = "select " + fields + " from " + database + "." + table + " where "
        if condition.strip() != '':
            query += condition + " and "

        query_to_get = query + "sys_id >  " + str(minr) + " and sys_id <= " + str(maxr)
        data = self.query(query_to_get)
        data = data.rename(columns=str.lower)

        return data

    def execute_join_query(self, database, table1, table2, fields1, fields2, on1, on2, drop_duplicates=True,
                           join='inner', verbose=False, block_size=10000, threads=70):
        temp_database = "dare_temp"
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
        temp_table = "temp_join_" + table1 + "_" + table2 + "_" + timestamp
        query = self.create_join_query(database, table1, table2, fields1, fields2, on1, on2, drop_duplicates, join)
        query = "create table " + temp_database + "." + temp_table + " as (" + query + ")"
        if verbose:
            display(query)
        # Execute join and store it in a temporal table
        res = self.execute_query(query)

        # Select from temporal table
        df = self.execute_select_query(temp_database, temp_table, "*", condition='', block_size=block_size,
                                       verbose=verbose, threads=threads)

        # Drop temporal table
        drop_query = "drop table " + temp_database + "." + temp_table
        drop_res = self.execute_query(drop_query, rds_query_lambda_name=rds_query_lambda_name, verbose=verbose,
                                      response=False)
        return df

    def execute_write_query(self, query):
        return self.execute_query(query)

    def execute_write_df(self, df, schema, table):
        print('data being written to snowflake db:', '\n', df.head())
        print(f'writing to {schema}, {table}')

        success, nchunks, nrows, _ = write_pandas(self.ctx, df, table, schema=schema, quote_identifiers=False)
        print(f"Writing is {success}, with {nchunks} and {nrows} rows")

    def create_table_from_df(self, s3_bucket, df, schema, table, drop_table=True, chunksize=30000, verbose=False,
                             parallel=True, force_drop_table=False, delete_table=False, delta=60):

        if "iopsredshiftdb" in schema:
            schema = schema.split(".")[1]  # iopsredshiftdb.dare_pre_hm
        else:
            schema = schema

        # truncate existing tables
        if drop_table:
            self.cs.execute(f"truncate table {schema}.{table};")
        if force_drop_table:
            self.cs.execute(f"truncate table {schema}.{table};")
        if delete_table:
            # deletes newest records of the destination table so new ones can be uploaded
            today = datetime.today().strftime('%Y-%m-%d')
            day_limit = str(pd.to_datetime(today) - pd.to_timedelta(delta, unit='d'))[0:10]
            day_limit = int(day_limit.replace('-', ''))
            self.cs.execute(f"delete from {schema}.{table} where day > {str(day_limit)};")

        print(f"Writing to {schema}.{table}")
        #         print(f"CREATE TABLE IF NOT EXISTS {schema}.{table};")
        #         self.cs.execute(f"CREATE TABLE IF NOT EXISTS {schema}.{table};")  # create the table
        self.execute_write_df(df, schema, table)
