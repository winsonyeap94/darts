# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:26:30 2020

@author: alvaro.garcia.piquer
"""

import pandas as pd
import numpy as np
import data_acquisition as da
import datetime
from dateutil.relativedelta import relativedelta
import pycountry_convert as pc


def process_field_rules(rules_str):
    # rules_str = "(c1,c2,c3) = ('V', '', '') | (c1,c2,c3) = ('A', '', '')"

    rules = [x.strip() for x in rules_str.split("|")]

    rule_list = []
    for r in rules:
        rule_dict = {}
        rule = [x.strip().replace('(', '').replace(')', '').replace("'", "").replace('"', "").split(",") for x in
                r.split("=")]
        for i in range(len(rule[0])):
            rule_dict[rule[0][i].lower()] = rule[1][i]
        rule_list.append(rule_dict)

    return rule_list


def get_rules(stock_type, md_stock_classification):
    # this function gets all of the stock classification rules from the md_stock_classification
    cat = stock_type[:-4]  # remove _qty or _val
    raw_rules = md_stock_classification[md_stock_classification['stock_classification'] == cat]
    cols = ['special_stock_indicator', 'stock_type', 'stock_category']
    raw_rules[cols] = raw_rules[cols].astype(str).replace('nan', '')
    rules_list = []
    for index, row in raw_rules.iterrows():
        rules_list = rules_list + [{'indspecstk': row['special_stock_indicator'],
                                    'stocktype': row['stock_type'],
                                    'stockcat': row['stock_category']}]
    return rules_list


def add_other_stk_rule(transformation_rules_table, df):
    # Find all the existing rules in Transformation for the STK rules
    t = transformation_rules_table[
        transformation_rules_table['Transformation'].str.contains("INDSPECSTK,STOCKTYPE,STOCKCAT")]["Transformation"]
    a_list = []
    a_list.extend(t.str.split("|").tolist())
    flat_list = [item.strip() for sublist in a_list for item in sublist]
    existing_rules = list(set(flat_list))
    df['indspecstock'] = df['indspecstk'] + "@" + df['stocktype'] + "@" + df['stockcat']
    rule = ""
    # From all the existing combinations in df, find the ones that have not been defined in Transformation
    for v in df['indspecstock'].unique():
        s = v.split('@')
        r = "(INDSPECSTK,STOCKTYPE,STOCKCAT) = ('" + s[0] + "','" + s[1] + "','" + s[2] + "')"
        if r not in existing_rules:
            rule += r + "|"
    # Remove last or symbol (|)
    rule = rule[:-1]
    df = df.drop(['indspecstock'], axis=1)
    transformation_rules_table.loc[transformation_rules_table['Transformation'] == 'other', 'Transformation'] = rule
    return transformation_rules_table


def add_untracked_other_categories(md_stock_classification, df):
    # this function adds to md_stock_classification classes than have not been identified as 'other' stock

    cols = ['special_stock_indicator', 'stock_type', 'stock_category']
    md_stock_classification[cols] = md_stock_classification[cols].astype(str).replace('nan', '')
    md_stock_classification['combi'] = md_stock_classification[cols].apply(lambda row: '_'.join(row.values.astype(str)),
                                                                           axis=1)

    cols = ['indspecstk', 'stocktype', 'stockcat']
    df2 = df[cols].drop_duplicates()
    df2['combi'] = df2[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    other_combis = df2[~df2.combi.isin(md_stock_classification['combi'])]
    other_combis = other_combis.rename(
        columns={'indspecstk': 'special_stock_indicator', 'stocktype': 'stock_type', 'stockcat': 'stock_category'})
    other_combis['stock_classification'] = 'other_stock'

    md_stock_classification = md_stock_classification.append(other_combis)

    return md_stock_classification


def process_special_locations(special_locations, df):
    data = df.copy()

    special_locations = special_locations.astype({'stock_category': 'str'})

    print(special_locations.columns)

    for index, row in special_locations.iterrows():
        print(index, len(special_locations))
        special_locations_flag = (data['plant_id'] == row['plant_id']) & (data['sloc_id'] == row['sloc_id'])
        cols = data.columns
        # get qty and val columns except the ones that match the stock_category column ("blocked" or "in_quality")
        # cols_qty = [c for c in cols if c.endswith("qty") and c != row['stock_category']+"_qty"]
        # cols_val = [c for c in cols if c.endswith("val") and c != row['stock_category']+"_val"]

        cols_qty = [c for c in cols if
                    c.endswith("qty") and c != row['stock_category'] + "_qty" and not c.startswith('vendor')]
        cols_val = [c for c in cols if
                    c.endswith("val") and c != row['stock_category'] + "_val" and not c.startswith('vendor')]
        data[cols_qty] = data[cols_qty].fillna(0)
        data[cols_val] = data[cols_val].fillna(0)
        # assign to the appropiate stock_category the sum of all the other categories
        data.loc[special_locations_flag, row['stock_category'] + "_qty"] = data.loc[
            special_locations_flag, cols_qty].sum(axis=1)
        data.loc[special_locations_flag, row['stock_category'] + "_val"] = data.loc[
            special_locations_flag, cols_val].sum(axis=1)
        data.loc[special_locations_flag, cols_qty] = 0
        data.loc[special_locations_flag, cols_val] = 0

    return data


def process_rules(transformation_rules, data, md_stock_classification):
    rules_dict = {}
    new_data = pd.DataFrame()

    for index, row in transformation_rules.iterrows():
        try:
            dest = row['Field'].split("/")[-1].lower()
            orig = row['Source Field'].split("/")[-1].lower()
        except Exception as e:
            print("data rules -> process_rules function, dataframe error:", e)
        if row['Transformation'] == 'drop' or row['Transformation'] == 'drop_after':
            continue

        elif row['Transformation'].startswith("value"):
            value = row['Transformation'].split("=")[1].strip().replace('"', '').replace("'", "")
            if row['Data type'] in ["QUAN", "CURR"]:
                try:
                    new_data[dest] = value.astype(float)
                except:
                    new_data[dest] = value
            else:
                new_data[dest] = value

        elif row['Transformation'] == 'map':
            try:
                new_data[dest] = data[orig]
            except KeyError:
                display("Column " + orig + " is missing")

    # First all the rows are created with map, and then we apply the transformations to the existing rows
    for index, row in transformation_rules.iterrows():
        if row['Transformation'] != 'map' and row['Transformation'] != 'drop' and row[
            'Transformation'] != 'drop_after' and not row['Transformation'].startswith("value"):
            dest = row['Field'].split("/")[-1].lower()
            orig = row['Source Field'].split("/")[-1].lower()
            # rule_list = process_field_rules(row['Transformation'])
            rule_list = get_rules(row['Field'], md_stock_classification)
            data_filt = data.copy()
            # Only transform the rows that fulfill at least one of the rules
            rule_columns = []
            for r in range(len(rule_list)):
                rule = rule_list[r]
                rule_col = 'use_tranf' + str(r)
                rule_columns.append(rule_col)
                data_filt[rule_col] = 1
                # If a rule field is not fulfilled, the rule is not accepted
                for k, v in rule.items():
                    try:
                        # print("k: "+k+", v: "+v+", data_filt:"+data_filt+", rule_col: "+rule_col)
                        data_filt.loc[data_filt[k] != v, rule_col] = 0
                    except:
                        # If column is not found, it may be due to upper case
                        data_filt.loc[data_filt[k.upper()] != v, rule_col] = 0
                # Rows to be transformed
                data_filt['use_tranf'] = data_filt[rule_columns].sum(axis=1)
                new_data.loc[data_filt['use_tranf'] > 0, dest] = data_filt.loc[data_filt['use_tranf'] > 0, orig]

    return new_data


def process_filters(filters, query_mode=True):
    filters_dict = {}

    for index, row in filters.iterrows():
        filters_dict[row['Source Field'].split("/")[-1]] = "(" + row['Filter'] \
            .replace("[", "").replace("]", "").replace('"', "'") + ")"

    if query_mode:
        filters_str = ""
        for k, v in filters_dict.items():  # INDSPECSTK (!= 'E')
            if v[1:3] == '!=':
                # print("DATA RULES ITEMS", k, v)  # (NOT INDSPECSTK = 'E' or INDSPECSTK is NULL)
                condition = " not in "
                v = f"(not {k} = {v[4:-1].strip()} or {k} is null)"  # "(" + v[4:].strip()
            else:
                condition = " in "
            if filters_str == "":
                filters_str += k + condition + v
            else:
                if condition == " in ":
                    filters_str += " and " + k + condition + v
                else:
                    #                     filters_str += " and " + k + condition + v
                    filters_str += " and " + v
        return filters_str

    return filters_dict


def get_fields_to_retrieve(transformation_rules_table):
    fields = []
    for index, row in transformation_rules_table.iterrows():
        try:
            dest = row['Field'].split("/")[-1].lower()
            orig = row['Source Field'].split("/")[-1].lower()
        except:
            orig = row['Source Field']
        if row['Transformation'] != 'drop':
            if orig not in fields:
                fields = fields + [orig]

    # fields_to_retrieve = "'"+"','".join(fields)+"'"

    return fields


def get_filtered_table(transformation_rules, db_engine, database, table_name, additional_condition="",
                       db_mode='redshift'):
    filters = transformation_rules[transformation_rules['Filter'].notnull()]
    filters_query = process_filters(filters, query_mode=True)
    if len(additional_condition.strip()) > 0:
        if len(filters_query.strip()) > 0:
            filters_query += " and "
        filters_query += additional_condition

    fields_to_retrieve = get_fields_to_retrieve(transformation_rules)

    # print(fields_to_retrieve, filters_query)
    if db_mode == 'redshift':
        df = db_engine.execute_select_query(database=database, table=table_name,
                                            fields=fields_to_retrieve, condition=filters_query)

    else:
        df = db_engine.execute_select_query(database=database, table=table_name,
                                            fields=fields_to_retrieve, condition=filters_query)

    if table_name != 'bw_stocks':
        pass
    else:
        df = df.astype(
            {
                'calyear': 'int32',
                'calmonth': 'int32',
                'calweek': 'int32',
                #             'calday': '',
                'stor_loc': 'str',
                'material': 'str',
                'plant': 'str',
                'mhmm_589': 'str',
                'indspecstk': 'str',
                'stocktype': 'str',
                'stockcat': 'str',
                'mhkz_532': 'float64',
                'mhkz_515': 'float64',
                'base_uom': 'str',
                'loc_currcy': 'str',
                'sys_id': 'int32'
            }
        )

        print(df.dtypes)

    return df


def process_numeric_columns(transformation_rules, data, db_mode):
    for index, row in transformation_rules.iterrows():
        if row['Data type'] in ['QUAN', 'CURR', 'NUMC'] and row['Transformation'] not in ['drop', 'drop_after']:
            field = row['Field']
            if db_mode in ['athena', 'mixed']:
                # store negative numbers
                # display(field)
                data[field] = treat_negatives(data, field)
            else:
                data[field] = data[field].astype(float)

    return data


def treat_negatives(data, field):
    negatives_vector = data[field].str.contains("-").fillna(False)
    if sum(negatives_vector) > 0:
        # convert emptys to nulls
        data[field] = data[field].replace(r'^\s*$', np.nan, regex=True)
        # remove negative sign and convert to float
        data[field] = data[field].str.replace("-", "").astype('float')
        # convert to negative
        data.loc[negatives_vector, field] = data.loc[negatives_vector, field] * (-1)
    else:
        data[field] = data[field].astype(float)

    return data[field]


def aggregate_by_pk(transformation_rules, data, sys_variables):
    treated_data = data.copy()
    numeric_fields = transformation_rules.loc[(transformation_rules['Transformation'] != 'drop') &
                                              (transformation_rules['Transformation'] != 'drop_after') &
                                              ((transformation_rules['Source Data type'] == 'QUAN') |
                                               (transformation_rules['Data type'] == 'CURR')), 'Field']
    treated_data[numeric_fields] = treated_data[numeric_fields].fillna(0)
    # sys_variables = ['sys_filename', 'sys_batch_id', 'sys_surrogate_key', 'sys_id']

    aggregations = {}
    for numeric_field in numeric_fields:
        aggregations[numeric_field] = 'sum'
    for sys_variable in sys_variables:
        aggregations[sys_variable] = 'max'

    # Group by
    pk_fields = transformation_rules.loc[transformation_rules.PK == "X", "Field"].to_list()
    pk_fields = [pk_field for pk_field in pk_fields if pk_field not in sys_variables]

    treated_data = treated_data.groupby(pk_fields)[numeric_fields.to_list() + sys_variables].agg(
        aggregations).reset_index()

    return treated_data


def join_external_data(bucket_name_additional_data, filename, drop_cols, join_on, data):
    external_data = da.read_excel_file(bucket_name_additional_data, 'external_files/master_data/' + filename + '.xlsx',
                                       tab=filename, skiprows=[], dtype=str).drop(columns=drop_cols)

    external_data = external_data.replace(np.nan, '', regex=True)

    treated_data = data.copy()
    treated_data = treated_data.merge(external_data, on=join_on, how='left')

    return treated_data


# Los inputs de esta función son:
# -->currency_exchange_data: el fichero csv donde tenemos las conversiones ('currency_conversion_rates/currency_conversion_rates.csv')
# --> data: el dataframe sobre el que queremos hacer la conversión
# --> currency_type_col: la columna del dataframe que nos dice que tipo de moneda tenemos
# --> value_col: la columna del dataframe que nos dice que cantidad (en moneda) tenemos
def currency_transformation(currency_exchange_data, data, currency_type_col, value_cols):
    import warnings

    data = data.copy()

    if value_cols == []:
        value_cols = [col for col in data.columns if '_val' in col]

    eur_data = {'iso': ['EUR'], 'rate': [1]}
    eur_data_df = pd.DataFrame(eur_data)
    currency_exchange_data = currency_exchange_data.append(eur_data_df)
    data = pd.merge(data, currency_exchange_data, how='left', left_on=currency_type_col, right_on='iso')
    data['rate'] = data['rate'].astype(float)

    for value_col in value_cols:
        data[value_col] = data[value_col].astype(float)
        data[value_col] = data[value_col] / data['rate']

    if (data['rate'].isnull().any().any()):
        warnings.warn('Careful, there is some mismatch between currency names !!!!')
    data[currency_type_col] = 'EUR'
    data = data.drop(['iso', 'rate'], axis=1)
    return data


def get_last_snapshot(table, key_columns, snapshot_date, date_column_name):
    """
    Get last snapshot from a table that only has records when a value is updated

    Args:
        table: snapshots table
        key_columns: keys of the table
        snapshot_date: date to obtain the snapshot
        date_column_name: dataframe with all the actions
    """

    # filter only past data in reference to the snapshot_date
    table = table[table[date_column_name] <= snapshot_date]

    # obtain max month by key
    max_month_by_key = table.groupby(key_columns)[date_column_name].max().reset_index()

    # left join values to max month
    full_data = max_month_by_key.merge(table, on=key_columns + [date_column_name], how='left')

    # rename date columns
    full_data['last_record_date'] = full_data[date_column_name]
    full_data[date_column_name] = snapshot_date

    return full_data


def lists_intersection(li1, li2):
    # common elements
    return set(li1) & set(li2)


def lists_junction(li1, li2):
    # NOT IN L1 BUT IN L2
    return (list(list(set(li2) - set(li1))))


def full_lists_junction(li1, li2):
    # not in both ways
    return (list(list(set(li1) - set(li2)) + list(set(li2) - set(li1))))


def valuate_quantity_columns(transformation_rules_table, data, sys_variables):
    """
    Get stock table, group by key without sloc, calculated aggregated cogs, valuate quantities by sloc and type

    Args:
        data: stock table
        transformation_rules_table: transformations master data
    """

    transform_table = transformation_rules_table[(transformation_rules_table['Transformation'] != 'drop') &
                                                 transformation_rules_table['Transformation'] != 'drop_after']
    qty_fields = transform_table.loc[(transform_table['Source Data type'] == 'QUAN'), 'Field']
    val_fields = transform_table.loc[(transform_table['Data type'] == 'CURR'), 'Field']
    # sys_variables = ['sys_id']#['sys_filename', 'sys_batch_id', 'sys_surrogate_key', 'sys_id']

    # remove vendor as it is processed separately
    qty_vendor_flag = ["vendor" not in qty for qty in qty_fields]
    qty_fields = qty_fields[qty_vendor_flag]
    val_vendor_flag = ["vendor" not in val for val in val_fields]
    val_fields = val_fields[val_vendor_flag]

    numeric_fields = qty_fields.to_list() + val_fields.to_list()
    data[numeric_fields] = data[numeric_fields].fillna(0)

    aggregations = {}
    for numeric_field in numeric_fields:
        aggregations[numeric_field] = 'sum'

    # Find PKs and delete
    pk_fields = transformation_rules_table.loc[transformation_rules_table['PK'] == 'X', 'Field'].to_list()
    pk_fields = [pk_field for pk_field in pk_fields if pk_field not in sys_variables]
    del (pk_fields[pk_fields.index("sloc_id")])

    agg_data = data.groupby(pk_fields).agg(aggregations).reset_index()
    agg_data['total_qty'] = agg_data[qty_fields].fillna(0).sum(axis=1)
    agg_data['total_val'] = agg_data[val_fields].fillna(0).sum(axis=1)
    agg_data['cogs'] = (agg_data.total_val / agg_data.total_qty).round(3)
    agg_data['cogs'] = agg_data['cogs'].replace([np.inf, -np.inf], np.nan).fillna(0)

    data = data.merge(agg_data[pk_fields + ['cogs']], on=pk_fields, how='left')

    del agg_data

    # new_val_names = [val_field + '_calc' for val_field in val_fields]
    data[val_fields] = data[qty_fields].multiply(data["cogs"], axis="index").fillna(0)

    data.drop(columns=['cogs'], inplace=True)

    return data


def valuate_by_md_price(db_engine, df, vars_to_valuate_by_md_price):
    """
    Gets stock table, queries to get dare_matloc_md from pre_hm, gets price, valuates columns provided in list by the matloc price

    Args:
        db_engine: connection engine Redshift or Athena
        df: stock dataframe
        vars_to_valuate_by_md_price: list of columns to valuate
    """

    matloc_md = db_engine.execute_select_query("dare_pre_hm", "dare_material_location_md",
                                               "material_id,plant_id,standard_price,price_units")

    matloc_md['price'] = matloc_md.standard_price / matloc_md.price_units
    matloc_md = matloc_md.replace([np.inf, -np.inf], np.nan)

    df = df.merge(matloc_md[['material_id', 'plant_id', 'price']], on=['material_id', 'plant_id'], how='left')

    for var_to_valuate in vars_to_valuate_by_md_price:
        df[var_to_valuate + '_val'] = df[var_to_valuate + '_qty'] * df.price
    df = df.drop(columns=['price'])

    try:
        df = df.astype(
            {
                'unrestricted_qty': 'float64',
                'in_quality_qty': 'float64',
                'in_transit_qty': 'float64',
                'blocked_qty': 'float64',
                'project_stock_qty': 'float64',
                'vendor_consignment_qty': 'float64',
                'customer_consignment_qty': 'float64',
                'in_transit_stock_transfer_qty': 'float64',
                'other_stock_qty': 'float64',
                'unrestricted_val': 'float64',
                'in_quality_val': 'float64',
                'in_transit_val': 'float64',
                'blocked_val': 'float64',
                'project_stock_val': 'float64',
                'vendor_consignment_val': 'float64',
                'customer_consignment_val': 'float64',
                'in_transit_stock_transfer_val': 'float64',
                'other_stock_val': 'float64'
            }
        )
    except Exception as e:
        print(e)

    return df


def remove_slocs_w_zero_qty(transformation_rules_table, df, sys_variables):
    """
    Gets stock table, removes all slocs that have 0 qty but always leaves 1 so the mat-loc is not lost

    Args:
        transformation_rules_table: transformations master data (ETL Design)
        df: stock dataframe
    """

    # get PK fields
    # sys_variables = ['sys_filename', 'sys_batch_id', 'sys_surrogate_key', 'sys_id']
    key_cols = transformation_rules_table.loc[transformation_rules_table['PK'] == 'X', 'Field'].to_list()
    key_cols = [key_col for key_col in key_cols if key_col not in sys_variables]
    del (key_cols[key_cols.index("sloc_id")])

    # key_cols = ['year', 'month', 'week', 'day', 'material_id', 'plant_id', 'pso', 'uom', 'currency']
    qty_cols = [col for col in df.columns if "_qty" in col]

    df_zero = df[df[qty_cols].sum(axis=1) == 0]
    df_non_zero = df[df[qty_cols].sum(axis=1) != 0]

    df_zero = df_zero.sort_values('sys_id', ascending=False).drop_duplicates(subset=key_cols)
    df_zero['sloc_id'] = ""

    df = df_non_zero.append(df_zero)

    return df


def get_region_from_country(new_df, country_field, region_field):
    regions = []

    for index, row in new_df.iterrows():
        try:
            regions += [pc.country_alpha2_to_continent_code(row[country_field])]
        except:
            regions += ['None']

    new_df[region_field] = regions
    new_df.loc[new_df[country_field] == 'DN', region_field] = 'AS'

    continents = {
        'NA': 'Americas',
        'SA': 'Americas',
        'AS': 'Asia-Pacific',
        'OC': 'Asia-Pacific',
        'AF': 'Asia-Pacific',  # to match MH classification
        'EU': 'Europe'
    }

    new_df.replace({'region': continents}, inplace=True)

    return new_df


def get_last_n_eoms(today, n):
    current_eom = datetime.datetime(today.year, today.month + 1, 1) - datetime.timedelta(days=1)

    eoms = []
    for lag in range(1, 13):
        eom = [str((current_eom + relativedelta(months=-lag)).date()).replace("-", "")]
        eoms = eoms + eom

    return eoms


def add_sys_variables(transformation_rules_table, table_name, sys_variables):
    sys_table = pd.DataFrame(columns=["Source Field", "Transformation", "Table", "Field"])
    # sys_vars = ["sys_id"]#["sys_filename", "sys_batch_id", "sys_surrogate_key", "sys_id"]
    sys_table['Source Field'], sys_table['Field'] = sys_variables, sys_variables
    sys_table['Transformation'] = ["map"] * len(sys_variables)
    sys_table['Table'] = [table_name] * len(sys_variables)
    sys_table['PK'] = ["X"] * len(sys_variables)
    transformation_rules_table = transformation_rules_table.append(sys_table, ignore_index=True)

    return transformation_rules_table


def get_data_structure_info(transformation_rules_table):
    columns_order = transformation_rules_table[
        (~transformation_rules_table.Field.isna()) & (transformation_rules_table.Transformation != 'drop') & (
                transformation_rules_table.Transformation != 'drop_after')].Field.to_list()

    cols_to_str = transformation_rules_table[
        (~transformation_rules_table.Field.isna()) & (transformation_rules_table.Transformation != 'drop') & (
                transformation_rules_table.Transformation != 'drop_after') & (
                transformation_rules_table['Redshift data type'] == 'varchar')].Field.to_list()

    cols_to_int = transformation_rules_table[
        (~transformation_rules_table.Field.isna()) & (transformation_rules_table.Transformation != 'drop') & (
                transformation_rules_table.Transformation != 'drop_after') & (
                transformation_rules_table['Redshift data type'] == 'int4')].Field.to_list()

    cols_to_float = transformation_rules_table[
        (~transformation_rules_table.Field.isna()) & (transformation_rules_table.Transformation != 'drop') & (
                transformation_rules_table.Transformation != 'drop_after') & (
                transformation_rules_table['Redshift data type'] == 'float4')].Field.to_list()

    return columns_order, cols_to_str, cols_to_int, cols_to_float


def apply_data_structure(columns_order, cols_to_str, cols_to_int, cols_to_float, df):
    df[cols_to_str] = df[cols_to_str].astype(str)
    df[cols_to_int] = df[cols_to_int].astype(int)
    df[cols_to_float] = df[cols_to_float].astype(float)
    df = df[columns_order]

    return df


def valuate_zero_val_column(movements, mat_loc, val_column):
    mat_loc['matloc_price'] = mat_loc.standard_price / mat_loc.price_units

    key = ['material_id', 'plant_id']
    movements = movements.merge(mat_loc[key + ['matloc_price']], on=key, how='left')

    movements['matloc_val'] = movements.matloc_price * movements.movement_qty
    movements[val_column] = np.where(movements[val_column] == 0, movements['matloc_val'], movements[val_column])

    movements = movements.drop(columns=['matloc_price', 'matloc_val'])

    return movements
