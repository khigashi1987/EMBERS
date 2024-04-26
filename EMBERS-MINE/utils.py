import json
import numpy as np
import pandas as pd
from io import StringIO
import base64
from datetime import time, datetime, date
from decimal import Decimal
from json import JSONEncoder
from config import Config

class MyJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, time):
            return obj.strftime('%H:%M:%S')
        elif isinstance(obj, Decimal):
            return float(obj) 
        elif isinstance(obj, pd.Series):
            return obj.to_list()
        elif isinstance(obj, np.int64):
            return int(obj) 
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):
            return base64.b64encode(obj).decode('utf-8')
        elif isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        return JSONEncoder.default(self, obj)

def make_info_project(out_prefix, project_json):
    if project_json is not None:
        project_json = project_json
    else:
        project_json = {}
    with open(f'{out_prefix}_project.json', 'w') as f:
        json.dump(project_json, f, indent=4)

def add_info_project(out_prefix, key, value):
    with open(f'{out_prefix}_project.json', 'r') as f:
        data = json.load(f)
    data[key] = value
    with open(f'{out_prefix}_project.json', 'w') as f:
        json.dump(data, f, indent=4)

def make_info_methods(out_prefix, methods_json):
    if methods_json is not None:
        methods_json = methods_json
    else:
        methods_json = {}
    with open(f'{out_prefix}_methods.json', 'w') as f:
        json.dump(methods_json, f, indent=4)

def make_sample_list(out_prefix, sample_df):
    if sample_df is not None:
        records_list = sample_df.to_dict(orient='records')
    else:
        records_list = []
    with open(f'{out_prefix}_samples.json', 'w') as f:
        json.dump(records_list, f, indent=4)
    with open(f'{out_prefix}_samples_update.json', 'w') as f:
        json.dump(records_list, f, indent=4)

def transpose_clean(df):
    df = df.transpose()
    df.reset_index(inplace=True)
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    return df

def match_sample_ID(target_id, reference_id):
    if type(target_id) is not str or type(reference_id) is not str:
        target_id = str(target_id)
        reference_id = str(reference_id)
    ID_patterns = []
    ID_patterns.append(target_id)
    ID_patterns.append(target_id.strip())
    #ID_patterns.append(target_id.split('_')[0])
    #ID_patterns.append(target_id.split('_')[1])
    ID_patterns.append(target_id.replace(' ', '_'))
    ID_patterns.append(target_id.replace(' ', ''))
    ID_patterns.append(target_id.replace(' ', '-'))
    ID_patterns.append(target_id.replace('_', ' '))
    ID_patterns.append(target_id.replace('-', ' '))
    if reference_id in ID_patterns:
        return True
    elif reference_id.split('_')[0] in ID_patterns:
        return True
    else:
        return False

def most_found_in_list(target_IDs, reference_IDs):
    # most of target_IDs can be found in reference_IDs
    # Threshold is 0.8
    found_count = 0
    for target_id in target_IDs:
        found = False
        for reference_id in reference_IDs:
            if match_sample_ID(target_id, reference_id):
                found = True
                found_count += 1
                break
        if not found:
            continue
    if found_count / len(target_IDs) > 0.8:
        return True
    else:
        return False

def check_table(content, sample_list):
    # check which column is the sample ID

    if len(content) < 5 and len(sample_list) > 5:
        # too few rows in content compared with sample_list length
        return None, None

    for i, c in enumerate(content.columns):
        values = content.iloc[:, i].dropna().values
        if len(values) == 0:
            continue
        # sampleID column should have the unique values for each row
        if len(set(list(values))) == len(values):
            # check most of content[c] values are in sample_list (list of dict) values
            for samplelist_key in sample_list[0].keys():
                samplelist_values = [d[samplelist_key] for d in sample_list]
                if most_found_in_list(values, samplelist_values):
                    content = content.set_index(c)
                    if most_found_in_list(content.index, content.columns):
                        # Skip this table because this seems to be all-vs-all comparison table
                        return None, None
                    # return the column name and samplelist_key of the matched IDs
                    return c, samplelist_key
    return None, None

def check_table_both_direction(out_prefix, content):
    sample_list = json.load(open(f'{out_prefix}_samples_update.json'))
    id_column, id_key = check_table(content, sample_list)
    if id_column is None and len(content) < 100:
        # transpose table if the size of table rows are not too large
        # with the assumption that the rows are metadata and columns are samples
        content = transpose_clean(content)
        id_column, id_key = check_table(content, sample_list)
    if type(id_column) is not str:
        return content, None, None
    if "Unnamed" in id_column or len(id_column) == 0:
        return content, None, None
    return content, id_column, id_key

def update_sample_list(out_prefix, content, id_column, id_key):
    sample_list = json.load(open(f'{out_prefix}_samples_update.json'))
    if id_column is None:
        return sample_list
    for i, c in enumerate(content.columns):
        if c == id_column:
            continue
        for j, row in content.iterrows():
            if pd.isna(row.iloc[i]):
                continue
            for d in sample_list:
                if match_sample_ID(row[id_column], d[id_key]):
                    d[c] = row[c]
        for d in sample_list:
            if c not in d:
                d[c] = None
    with open(f'{out_prefix}_samples_update.json', 'w') as f:
        f.write(json.dumps(sample_list, indent=4, cls=MyJSONEncoder))
    return sample_list
    
def bad_column_name(column_name):
    if column_name is None:
        return True
    if "Unnamed" in column_name or len(column_name) == 0:
        return True
    return False

def cleanse_sample_list(out_prefix):
    sample_list = json.load(open(f'{out_prefix}_samples_update.json'))
    DB_keys = Config.DATABASE_RELATED_KEYS

    # copy sample_list to sample_list_update
    sample_list_update = []
    for d in sample_list:
        sample = {}
        for k, v in d.items():
            if bad_column_name(k):
                continue
            new_k = k.strip()
            sample[new_k] = v
        sample_list_update.append(sample)

    all_keys = set()
    for d in sample_list_update:
        all_keys.update(d.keys())
    for d in sample_list_update:
        for key in all_keys:
            if key not in d:
                d[key] = None

    df = pd.DataFrame(sample_list_update)

    numerical_keys = []
    categorical_keys = []
    for column in df.columns:
        non_nan_values = df[column].dropna()

        try:
            numeric_values = pd.to_numeric(non_nan_values, errors='coerce')
            if numeric_values.notna().all():
                # 全ての値が数値に変換可能な場合
                df[column] = pd.to_numeric(df[column])
                if column not in DB_keys:
                    numerical_keys.append(column)
            else:
                # 数値に変換できない値が含まれている場合
                df[column] = df[column].fillna('')
                if column not in DB_keys:
                    categorical_keys.append(column)
        except Exception:
            # 例外が発生した場合はカテゴリカルとして扱う
            df[column] = df[column].fillna('')
            categorical_keys.append(column)

    sample_list_update = df.to_dict(orient='records')
    with open(f'{out_prefix}_samples_update.json', 'w') as f:
        f.write(json.dumps(sample_list_update, indent=4, cls=MyJSONEncoder))

    categorical_items = {}
    numerical_items = {}
    # From sample_list, extract unique values for each categorical key
    # and 5 not-NaN values for each numerical key
    for key in categorical_keys:
        categorical_items[key] = df[key].unique().tolist()
    for key in numerical_keys:
        numerical_items[key] = df[key].dropna().unique().tolist()[:5]

    return numerical_items, categorical_items

def update_project_schema(result_dict):
    schema = json.load(open(Config.SCHEMA_PROJECT_JSON))
    for k, value in result_dict.items():
        if k == 'country' or k == 'disease':
            for v in value:
                if v not in schema[k]:
                    schema[k].append(v)
    return schema

def current_keys(out_prefix):
    sample_list = json.load(open(f'{out_prefix}_samples_update.json'))
    DB_keys = Config.DATABASE_RELATED_KEYS

    all_keys = set()
    for d in sample_list:
        all_keys.update(d.keys())
    
    # copy all_keys to new_keys
    new_keys = all_keys.copy()
    for k in all_keys:
        if k in DB_keys:
            new_keys.remove(k)
    
    return list(new_keys)