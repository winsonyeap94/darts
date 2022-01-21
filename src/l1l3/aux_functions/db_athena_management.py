# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:28:56 2020

@author: alvaro.garcia.piquer
"""
import db_management

import pyathena
from pyathena.pandas.cursor import PandasCursor

ATHENA_PARAMETERS = {'resource_name': 's3', 'athena_bucket_name': 'dare-iops-athena-data', 'region_name': 'eu-central-1'}

class DBAthena(db_management.DB):



    def __init__(self,
                 resource_name=ATHENA_PARAMETERS['resource_name'], 
                 athena_bucket_name=ATHENA_PARAMETERS['athena_bucket_name'], 
                 region_name=ATHENA_PARAMETERS['region_name']):
        athena_bucket = ssm.get_parameter(Name='/bucketname/athena-data').get("Parameter").get("Value")
        self.resource_name = resource_name
        self.athena_bucket_name = athena_bucket
        self.region_name = region_name
    
    def connect(self):
        self.conn = pyathena.connect(s3_staging_dir=self.resource_name+"://"+self.athena_bucket_name+"/", region_name=self.region_name, cursor_class=PandasCursor)
        
        
        
        
        
        