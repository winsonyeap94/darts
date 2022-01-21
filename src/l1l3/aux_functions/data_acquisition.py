# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:03:27 2020

@author: alvaro.garcia.piquer
"""

import boto3
import pandas as pd
import os
from typing import Tuple

ATHENA_PARAMETERS = {
    "resource_name": "s3",
    "athena_bucket_name": "dare-iops-athena-data",
    "region_name": "eu-central-1",
}

ssm = boto3.client('ssm')


def serie_to_str_list(df, field):
    return [c.strip() for c in df[field]]


def obtain_headers(line):
    # The content of the query file is parsed to obtain the table name, the columns, the constraints and the primary keys
    table = line[0: line.find("(")].split('"')[1::2]
    table[1] = table[1].split("/")[-1]
    content = line[line.find("(") + 1: line.rfind(")")]
    cols_const_pk = content.split("PRIMARY KEY")
    cols_const = cols_const_pk[0]
    pk = cols_const_pk[1][cols_const_pk[1].find("(") + 1: cols_const_pk[1].find(")")]
    pk = pk.split('"')[1::2]
    cols_const = cols_const.split("CONSTRAINT")
    cols = cols_const[0].split('"')[1::2]
    const = cols_const[1].split('"')[1::2]
    return table, cols, const, pk


def obtain_bw_headers(line, table_prefix):
    # The content of the query file is parsed to obtain the table name and the columns
    lines = line.split("\r\n")
    table_lines = lines[2].split("\\")
    table = []
    table.append(table_lines[-2])
    table_name = table_prefix + table_lines[-1].replace(".csv", "").replace("DARE_", "")
    table_name = "".join([i for i in table_name if not i.isdigit()]).replace("__", "")
    table.append(table_name)
    pos = lines.index("COLUMN;FIELDNAME;KEY;TYPE;LENGTH;OUTPUTLEN;DECIMALS")
    lines = lines[pos + 1:]
    cols = []
    for element in lines:
        if len(element) > 0:
            cols.append(element.split(";")[1].split("/")[-1])
    return table, cols


def obtain_all_files(resource_name, bucket_name):
    # Obtain all files
    s3 = boto3.resource(resource_name)
    bucket = s3.Bucket(bucket_name)

    dict_folders = {}
    for obj in bucket.objects.all():
        if "sql" in obj.key:
            key = obj.key.split("/")[0]
            dict_folders[key] = {}
            body = obj.get()["Body"].read().decode("utf-8")
            table_name, columns, constraints, primary_keys = obtain_headers(body)
            dict_folders[key]["table_name"] = table_name
            dict_folders[key]["columns"] = columns
            dict_folders[key]["constraints"] = constraints
            dict_folders[key]["primary_keys"] = primary_keys
    return dict_folders


def obtain_bw_files(resource_name, bucket_name, folder_prefix, table_prefix):
    # Obtain all files
    s3 = boto3.resource(resource_name)
    bucket = s3.Bucket(bucket_name)

    dict_folders = {}
    for obj in bucket.objects.all():
        if obj.key.startswith(folder_prefix):
            if obj.key.split("/")[-1].startswith("S_"):
                key = obj.key.split("/")[0]
                if key.endswith("INITIAL"):
                    type_prefix = "initial_"
                elif key.endswith("HISTORICAL"):
                    type_prefix = "historical_"
                else:
                    type_prefix = ""
                dict_folders[key] = {}
                body = obj.get()["Body"].read().decode("utf-8")
                table_name, columns = obtain_bw_headers(
                    body, table_prefix + type_prefix
                )
                dict_folders[key]["table_name"] = table_name
                dict_folders[key]["columns"] = columns
    return dict_folders


def list_to_chunks(l, n):
    """Yield successive n-sized chunks from lst."""
    n = max(1, n)
    return list(l[i: i + n] for i in range(0, len(l), n))


def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client("s3")
    try:
        s3.upload_file(local_file, bucket, s3_file)
        # print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False


def convert_list_to_str(values):
    values = [str(v) for v in values]
    return "'" + "','".join(values) + "'"


def werks_from_country(db_engine, database, mandt, countries, to_str=True):
    werks = db_engine.execute_select_query(
        database=database,
        table="T001W",
        fields="distinct werks",
        condition="land1 in ("
                  + convert_list_to_str(countries)
                  + ") and mandt='"
                  + mandt
                  + "'",
    )
    if to_str:
        return convert_list_to_str(werks["werks"])
    else:
        return werks


def lifnr_from_country(
        db_engine, database, countries, mandt=None, no_werks="", to_str=True
):
    condition = ""
    if mandt is not None:
        condition += "and mandt='" + mandt + "' "
    if no_werks is not None:
        if condition != "":
            condition += "and "
        condition += "werks!='" + no_werks + "' "
    lifnr = db_engine.execute_select_query(
        database=database,
        table="LFA1",
        fields="distinct lifnr",
        condition="land1 in (" + convert_list_to_str(countries) + ") " + condition,
    )
    if to_str:
        return convert_list_to_str(lifnr["lifnr"])
    else:
        return lifnr


def read_excel_file(
        s3_bucket, file, tab, skiprows=[0], keep_default_na=True, dtype=str
):
    data_location = "s3://{}/{}".format(s3_bucket, file)
    df = pd.read_excel(
        data_location,
        tab,
        skiprows=skiprows,
        keep_default_na=keep_default_na,
        dtype=dtype,
        engine='xlrd'  # do not change! IT MUST BE XLRD
    )
    return df


def read_csv_file(
        s3_bucket, file, skiprows=[0], dtype=None, sep=";", header=None, usecols=None
):
    data_location = "s3://{}/{}".format(s3_bucket, file)
    df = pd.read_csv(
        data_location,
        skiprows=skiprows,
        dtype=dtype,
        sep=sep,
        header=header,
        error_bad_lines=False,
        usecols=usecols,
    )
    return df


def obtain_available_tables_db(db_engine, database, s3_template_bucket, template_file):
    all_tables_in_db = db_engine.execute_query("SHOW TABLES IN " + database)
    all_tables_in_db = serie_to_str_list(all_tables_in_db, "tab_name")
    template_tables = read_excel_file(
        s3_template_bucket, template_file, "Overview", skiprows=[0]
    )
    available_tables = [
        table for table in template_tables["Table"] if table.lower() in all_tables_in_db
    ]
    return available_tables


def save_df_to_db(
        df,
        s3_bucket,
        db_engine,
        database,
        table,
        temp_preceding_path="",
        sep=",",
        drop_table=True,
        folder_name=None,
):
    temporal_file_name = temp_preceding_path + "output/" + table + ".csv"
    if folder_name is None:
        folder_name = "output_" + table
    s3_file_name = "outputs/" + folder_name + "/" + table + ".csv"

    df_types = {}
    for c in df.columns:
        if df.dtypes[c] == "object":
            df_types[c] = str
        else:
            df_types[c] = df.dtypes[c]
    if not drop_table:
        # Incremental. Read the data stored in Athena to add the new rows
        df_prev = read_csv_file(s3_bucket, s3_file_name, skiprows=[], dtype=df_types)
        df = df_prev.append(df, ignore_index=True)
        df = df.drop_duplicates()

    # df = df.fillna(0)
    df.to_csv(temporal_file_name, index=False, sep=sep)
    uploaded = upload_to_aws(
        temp_preceding_path + "output/" + table + ".csv", s3_bucket, s3_file_name
    )
    if db_engine is not None:
        data_type = {}
        for c in df.columns:
            if df.dtypes[c] == "float64":
                data_type[c] = "double"
            elif df.dtypes[c] == "int64":
                data_type[c] = "int"
            else:
                data_type[c] = "string"
        db_engine.create_table(
            "s3",
            s3_bucket,
            database,
            "outputs/output_" + table,
            table,
            df.columns,
            contains_header=True,
            data_type=data_type,
        )
    if os.path.exists(temporal_file_name):
        os.remove(temporal_file_name)

    return s3_file_name, df


def get_additional_condition(
        db_engine, rejection_condition, write_database, dest_table
):
    # Additional condition to only select rows with a greater sys_id which should be the new ones

    # select environment and adequate lambda name
    environment = getenv()
    rds_query_lambda_name = get_redshift_query_lambda_name(environment)

    max_sys_id_query = "select max(sys_id) from " + write_database + "." + dest_table
    max_sys_id = db_engine.execute_query(
        max_sys_id_query, verbose=True, rds_query_lambda_name=rds_query_lambda_name
    )

    max_sys_id_str = max_sys_id[0].replace("[", "").replace("]", "")

    # max_sys_id_str = str(max_sys_id[0])
    additional_condition = "sys_id > " + max_sys_id_str + " and " + rejection_condition

    # max_sys_id = db_engine.execute_select_query(database=write_database, table=dest_table, fields='max(sys_id)', verbose=True, block_size=100000)
    # max_sys_id = max_sys_id.max()

    return additional_condition


""" 
    #Deltas done through calday or calweek. Deprecated.
    try:
        max_day = db_engine.execute_select_query(database=write_database, table=dest_table, fields='max(day)', verbose=False)
        max_day = max_day.max()[0].replace("-","")
        additional_condition = "calday > '" + max_day + "' and " + rejection_condition
        
    except:

        try:
            max_week = db_engine.execute_select_query(database=write_database, table=dest_table, fields='max(week)', verbose=False)
            max_week = max_week.max()[0].replace("-","")

            # careful: same week can be in two months --> temporary fix: drop_duplicates in save df to db
            #max_month = athena_db.execute_select_query(database=read_database, table=dest_table, 
            #     fields='max(month) as month', condition='week = 'max_week)['month'][0] #+ "and calmonth > '" + max_month +  "'"
            additional_condition = "calweek >= '" + max_week + "' and " + rejection_condition 
        except:
            additional_condition = rejection_condition
    
    return additional_condition"""


def getenv() -> str:
    aid = boto3.client("sts").get_caller_identity().get("Account")
    if aid == "944917888538":
        return "test"
    if aid == "196518573157":
        return "dev"
    if aid == "286321116228":
        return "prod"
    return "sandbox"


def get_buckets(environment: str) -> Tuple[str, str]:
    additional_data = ssm.get_parameter(Name='/bucketname/additional-data').get("Parameter").get("Value")
    raw_data = ssm.get_parameter(Name='/bucketname/raw-data').get("Parameter").get("Value")

    # bucket_name_additional_data = "dare-iops-additional-data-sandbox"
    # bucket_raw_data = "dare-iops-raw-data-sandbox"
    # if environment == "test":
    #     bucket_name_additional_data = "dare-iops-additional-data"
    #     bucket_raw_data = "dare-iops-raw-data"
    # elif environment == "dev":
    #     bucket_name_additional_data = "dare-iops-additional-data-dev"
    #     bucket_raw_data = "dare-iops-raw-data-dev"
    # elif environment == "prod":
    #     bucket_name_additional_data = "dare-iops-additional-data-prod"
    #     bucket_raw_data = "dare-iops-raw-data-prod"
    # print(bucket_name_additional_data, "is the additional data bucket")
    return additional_data, raw_data


def get_redshift_query_lambda_name(environment: str) -> str:

    return "redshiftquery"


def write_csv_file(
        s3_bucket,
        file,
        dataframe: pd.DataFrame,
        sep=";",
        decimal=".",
        index=False,
        mode="w",
        header=True,
):
    """
    Function to write a dataframe to a csv file from a S3 bucket
    """
    data_location = "s3://{}/{}".format(s3_bucket, file)
    print("writing to location", data_location)
    dataframe.to_csv(
        data_location, sep=sep, decimal=decimal, index=index, mode=mode, header=header
    )
    return None
