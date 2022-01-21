# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:22:52 2020

@author: alvaro.garcia.piquer
"""

class DB:
    
    def __init__(self):
        self.conn = None
        
    def connect(self):
        raise NotImplementedError
    
    def create_table(self, s3_bucket, s3_folder, database, key, table, columns, sep=",", contains_header=False, data_type=None, partitioned=None):
        query = "DROP TABLE IF EXISTS " + database + "." + table + ";"
        cur = self.conn.cursor()
        cur.execute(query)
        cur.close()
        
        if data_type is None:
            data_type = {}
            for c in columns:
                data_type[c] = 'string'
                
        serdeproperties = "ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' \
        WITH SERDEPROPERTIES ('serialization.format' = '"+ sep + "', 'separatorChar' = '" + sep + "', 'quoteChar' = '\"')"
            
        tbl_properties = "'has_encrypted_data'='false'"
        partitioned_field = ""
        if partitioned is not None:
            partitioned_field = "PARTITIONED BY (`"+ partitioned + "` " + data_type[partitioned] + ")"
        if contains_header:
            tbl_properties += ", 'skip.header.line.count'='1'"
        query = "CREATE EXTERNAL TABLE IF NOT EXISTS " + database + "." + table + " ("
        for c in columns:
            if c != partitioned:
                query += "`" + c + "` " + data_type[c] + ","
        query = query[0:-1]
        query += ") " + partitioned_field + " " + serdeproperties + " LOCATION '" + s3_bucket + '://' + s3_folder + '/' + key + "'" + \
            " TBLPROPERTIES (" + tbl_properties +");"
        cur = self.conn.cursor()
        cur.execute(query)
        cur.close()
        return query
    
    def create_view(self, database, view_name, view_query):
        query = "DROP VIEW IF EXISTS " + database + "." + view_name + ";"
        cur = self.conn.cursor()
        cur.execute(query)
        cur.close()
        
        query = "CREATE VIEW " + database + "." + view_name + " as " + view_query + ";"
        cur = self.conn.cursor()
        cur.execute(query)
        cur.close()
        return query
    
    def execute_query(self, query):
        cursor = self.conn.cursor()
        df = cursor.execute(query).as_pandas()
        return df
    
    def execute_select_query(self, database, table, fields, condition='', block_size=10000):
        if isinstance(fields, list):
            fields = [field for field in fields if str(field) != 'nan'] #remove NANs
            fields = ",".join(fields) # convert list format to appropiate query format
        query = "select " + fields + " from " + database + "." + table
        if condition.strip() != '':
            query += " where " + condition
        #display(query)
        #df = pd.read_sql(query, self.conn)
        cursor = self.conn.cursor()
        df = cursor.execute(query).as_pandas()
        return df
    
    def create_join_query(self, database, table1, table2, fields1, fields2, on1, on2, drop_duplicates=True, join='inner'):
        query = "select "
        if drop_duplicates:
            query += "distinct "
        fields = ''
        if len(fields1) > 0:
            fields += ("t1.")+(","+"t1.").join(fields1)
        if len(fields2) > 0:
            if len(fields1) > 0:
                fields += ','
            fields += ("t2.")+(","+"t2.").join(fields2)
        if len(fields)==0:
            fields = 't1.*, t2.*'
        query += fields + " from " +  database + "." + table1 + " as t1"
        query += " " + join + " join " +  database + "." + table2 + " as t2"
        on_comp = ""
        for i in range(0, len(on1)):
            on_comp += "t1."+on1[i]+"=t2."+on2[i]
            if i < (len(on1) - 1):
                on_comp += " and "
        query += " on " + on_comp
        display(query)                          
        return query
    
    def execute_join_query(self, database, table1, table2, fields1, fields2, on1, on2, drop_duplicates=True, join='inner'):
        query = self.create_join_query(database, table1, table2, fields1, fields2, on1, on2, drop_duplicates, join)
        cursor = self.conn.cursor()
        df = cursor.execute(query).as_pandas()
        return df
                
    def create_tables(self, resource_name, bucket_name, database, dict_folders, sep=",", partitioned={}):
        queries = []
        for key, values in dict_folders.items():
            try:
                if key not in partitioned:
                    part = None
                else:
                    part = partitioned[key]
                query = self.create_table(resource_name, bucket_name, database, key, values["table_name"][1], values["columns"], sep, partitioned=part)
                queries.append(query)
            except:
                display("Error exporting from S3 to Athena:", key)
        return queries
            
