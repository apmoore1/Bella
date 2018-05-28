import os
import json

import numpy as np
import pandas as pd

def get_json_data(result_path, dataset_name):
    if os.path.isfile(result_path):
        with open(result_path, 'r') as result_file:
            data_dict = json.load(result_file)
            if dataset_name in data_dict:
                return data_dict[dataset_name]
    return {}

def write_json_data(result_path, dataset_name, data):
    data_dict = {}
    if os.path.isfile(result_path):
        with open(result_path, 'r') as result_file:
            data_dict = json.load(result_file)
    data_dict[dataset_name] = data
    with open(result_path, 'w') as result_file:
        json.dump(data_dict, result_file)

def get_pandas_data(result_path, column, index, index_column):
    if os.path.isfile(result_path):
        with open(result_path, 'r') as result_file:
            pd_data = pd.read_csv(result_file)
            pd_data = pd_data.set_index(index_column)
            data = pd_data[str(column)][index]
            if data != 0:
                return data
def save_pandas_data(result_path, column, index, index_column, data):
    if os.path.isfile(result_path):
        with open(result_path, 'r') as result_file:
            pd_data = pd.read_csv(result_file)
            pd_data = pd_data.set_index(index_column)
        pd_data.loc[index, str(column)] = data
        with open(result_path, 'w') as result_file:
            pd_data.to_csv(result_file)
def create_pandas_file(result_path, columns, index_key, index_values, re_write=False):
    if os.path.isfile(result_path) and not re_write:
        return
    pd_data = pd.DataFrame(np.zeros((len(index_values), len(columns))), columns=columns)
    pd_data[index_key] = index_values
    pd_data = pd_data.set_index(index_key)
    with open(result_path, 'w') as result_file:
        pd_data.to_csv(result_file)
