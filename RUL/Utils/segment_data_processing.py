import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob as gl
from utils import read_data
from tqdm.notebook import tqdm


def split_sensor_id(col_name):
    return col_name.split('_')[1]

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns[3:]:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df



def select_assets(DATA_DIR, error_code_label='error_code_53', min_segment_length=7, max_segment_length=50):
    """
    This function select asset that satisfied with these conditions:
        - Must have data of sensors with ids in (17,34,29,117)
        - Nan values for each columns must be < 30%

    Params:
        DATA_DIR (list): List of path to asset data folders
        max_segment_length (int): Max length for each error segments
    Return:
        select_ids (list): paths to selected asset data file
    """
    
    select_ids = []
    for i in tqdm(range(len(DATA_DIR)), desc='Selecting Assets'):
        file_path = os.path.join(DATA_DIR[i], 'day_grouped.csv')
        data = read_data(file_path, error_code_label)
        
        # get indexes of error codes
        indexes = np.argwhere(np.array(data.label == 1)).flatten()

        # select indexes between 2 range 
        selected_index = np.where(np.logical_and(np.diff(indexes) >= min_segment_length, np.diff(indexes) <= max_segment_length))[0]
        if len(selected_index) < 0:
            continue

        # check stdev, if min or max value of a sensor is Nan 
        # -> its stdev must be nan also
        for col in data.columns:
            if 'stdev' not in col:
                continue
            sensor_id = split_sensor_id(col)
            max_col = f'max_{sensor_id}'
            data.loc[data[max_col].isnull(), col] = np.nan

        # drop all nan or 30 % nan
        data.dropna(how='all', axis=1, inplace=True)
        data.dropna(axis=1, thresh=int(data.shape[0]*0.7), inplace=True)

        # drop columns with 1 value
        nunique = data.nunique()
        cols_to_drop = nunique[nunique == 1].index
        data.drop(cols_to_drop, axis=1, inplace=True)
    
        # get sensor ids
        columns = data.select_dtypes(include=['float64']).columns.values
        sensor_ids = set(map(lambda i:split_sensor_id(i), columns))

        if len(sensor_ids.intersection({'17', '34', '29', '117'})) == 4:
            select_ids.append(file_path)
    return select_ids


def select_and_merge(selected_assets, error_code_label='error_code_53'):
    col_count = dict()
    all_data = []
        
    for i in tqdm(range(len(selected_assets)), desc='Reading assets'):
        data = read_data(selected_assets[i], error_code_label)
        all_data.append(data)
        
        columns = data.columns.values
        # count appearances of each columns
        for col in columns:
            if col not in col_count:
                col_count[col] = 1
            else:
                col_count[col] = col_count[col]+1
                
    num_selected_assets = len(selected_assets)
#     selected_columns = []
#     for col_name, count in col_count.items():
#         # only select columns with 50% or above not nan
#         if count > num_selected_assets*0.5:
#             selected_columns.append(col_name)
#     selected_columns.append('set_AssetId')
#     selected_columns.append('set_RfrErrorCode')

    print('Start merging and processing')
    
    # merge files and reduct memory 
    merged_data = pd.concat(all_data.copy(), ignore_index=True)
    merged_data = reduce_mem_usage(merged_data)
    del all_data

    # remove nan and select columns
#     merged_data.dropna(how='all', inplace=True)
#     merged_data.dropna(axis=1, thresh=int(merged_data.shape[0]*0.65), inplace=True)
#     merged_data.drop(merged_data.columns[~merged_data.columns.isin(selected_columns)], axis=1, inplace=True)

    drop_cols = []
    for col in merged_data.columns:
        if not 'stdev' in col:
            continue
        sensor_id = split_sensor_id(col)
        max_col = f'max_{sensor_id}'

        # check invalid stdev = 0 sensor ids
        if max_col not in merged_data.columns and sensor_id.isdigit():
            drop_cols.append(col)
        else:
            merged_data.loc[merged_data[max_col].isnull(), col] = np.nan
    merged_data.drop(drop_cols, axis=1, inplace=True)
    
    return merged_data
    

def select_columns(selected_assets, error_code_label='error_code_53'):
    """
    This function select columns appeared in at least 50% numbers of selected assets

    Params:
        selected_assets (list): paths to selected asset data file
    Return:
        select_ids (list): selected column names
    """
    
    col_count = dict()
    for i in tqdm(range(len(selected_assets)), desc='Selecting Columns'):
        data = read_data(selected_assets[i], error_code_label)
        columns = data.columns.values
        # count appearances of each columns
        for col in columns:
#             if 'list' in col or 'set' in col:
#                 continue
            if col not in col_count:
                col_count[col] = 1
            else:
                col_count[col] = col_count[col]+1
                
    num_selected_assets = len(selected_assets)
    selected_columns = []
    for col_name, count in col_count.items():
        # only select columns with 50% or above not nan
        if count > num_selected_assets*0.5:
            selected_columns.append(col_name)
    selected_columns.append('set_AssetId')
    selected_columns.append('set_RfrErrorCode')
 
    return selected_columns


def merge_and_process(selected_assets, selected_columns, error_code_label='error_code_53'):
    """
    This function merge selected assets and columns to a single frame

    Params:
        selected_assets (list): paths to selected asset data file
        selected_columns (list): selected columns name

    Return:
        merged_data (DataFrame): merged data
    """
    
    merged_data = pd.DataFrame()
    
    # read all data files
    all_data = []
    for i in tqdm(range(len(selected_assets)), desc='Reading assets'):
        data = read_data(selected_assets[i], error_code_label)
        all_data.append(data)
        
    print('Start merging and processing')
    
    # merge files and reduct memory 
    merged_data = pd.concat(all_data.copy(), ignore_index=True)
    merged_data = reduce_mem_usage(merged_data)
    del all_data

    # remove nan and select columns
    merged_data.dropna(how='all', inplace=True)
    merged_data.dropna(axis=1, thresh=int(merged_data.shape[0]*0.65), inplace=True)
    merged_data.drop(merged_data.columns[~merged_data.columns.isin(selected_columns)], axis=1, inplace=True)

    drop_cols = []
    for col in merged_data.columns:
        if not 'stdev' in col:
            continue
        sensor_id = split_sensor_id(col)
        max_col = f'max_{sensor_id}'

        # check invalid stdev = 0 sensor ids
        if max_col not in merged_data.columns and sensor_id.isdigit():
            drop_cols.append(col)
        else:
            merged_data.loc[merged_data[max_col].isnull(), col] = np.nan
    merged_data.drop(drop_cols, axis=1, inplace=True)
    
    return merged_data


def get_segments(merged_data, min_error_day, max_error_day=None):
    """
    This function get segments data from merged data, error code is selected based on label column

    Params:
        merged_data (DataFrame): merged assets data
        min_error_day, max_error_day (int): min and max values of days between 2 errors (segment length) to select

    Return:
        segments_data (DataFrame): segments data with labeled segments
    """
    
    segments_data = []
    unit_counts = 1
    print('Start extracting segments')
    for asset_id in tqdm(merged_data['set_AssetId'].unique()):
        # for each asset, drop nan columns and fill missing values with forward fill    
        asset_data = merged_data[merged_data['set_AssetId'] == asset_id].copy().reset_index(drop=True)
        asset_data.dropna(how='all', inplace=True)
        asset_data.interpolate(method='ffill', inplace=True)  

        # get indexes of error codes
        indexes = np.argwhere(np.array(asset_data.label == 1)).flatten()
        indexes = np.insert(indexes, 0, 0)
        
        # select indexes between 2 range 
        if max_error_day is None:
            selected_index = np.where(np.diff(indexes) >= min_error_day)[0]
        else:
            selected_index = np.where(np.logical_and(np.diff(indexes) >= min_error_day, np.diff(indexes) <= max_error_day))[0]

        for index in selected_index:
            start_index = indexes[index]
            if start_index != 0:
                start_index += 1
            
            stop_index = indexes[index+1]
            segment = asset_data.iloc[start_index:stop_index].copy()
            segment['Timestep'] = np.arange(1, len(segment)+1) 
            
            choose_segment = True
            for col in segment.columns: 
                # double check null values
                if segment[col].isnull().sum() != 0: 
                    choose_segment = False

            if choose_segment:
                segment['Unit'] = unit_counts
                segments_data.append(segment)
                unit_counts += 1
    segments_data = pd.concat(segments_data)
    segments_data.drop(['label'], axis=1, inplace=True)
    segments_data.reset_index(drop=True, inplace=True)
    
    return segments_data

def get_segments_optimized(merged_data, min_error_day, max_error_day=None):
    """
    This function get segments data from merged data, error code is selected based on label column

    Params:
        merged_data (DataFrame): merged assets data
        min_error_day, max_error_day (int): min and max values of days between 2 errors (segment length) to select

    Return:
        segments_data (DataFrame): segments data with labeled segments
    """
    
    segments_data = []
    unit_count = 1
    print('Start extracting segments')
    for asset_id in tqdm(merged_data['set_AssetId'].unique()):
        # for each asset, drop nan columns and fill missing values with forward fill    
        asset_data = merged_data[merged_data['set_AssetId'] == asset_id].copy().reset_index(drop=True)
        asset_data.dropna(how='all', inplace=True)
        asset_data.interpolate(method='ffill', inplace=True)  

        # get indexes of error codes
        indexes = np.argwhere(np.array(asset_data.label == 1)).flatten()
        indexes = np.insert(indexes, 0, 0)
        
        # select indexes between 2 range 
        if max_error_day is None:
            selected_index = np.where(np.diff(indexes) >= min_error_day)[0]
        else:
            selected_index = np.where(np.logical_and(np.diff(indexes) >= min_error_day, np.diff(indexes) <= max_error_day))[0]
        
        if len(selected_index) == 0:
            continue
        # create index range
        segment_index = []
        for index in selected_index:
            start_index = indexes[index]
            if start_index != 0:
                start_index += 1

            stop_index = indexes[index+1]
            segment_index.append(range(start_index, stop_index))
        
        # create new columns with units label assigned to selected index
        unit_indexes = []
        for number, segment in enumerate(segment_index):
            for index in segment:
                unit_indexes.append([number+unit_count, index])

        unit_indexes = np.array(unit_indexes)
        unit_indexes = pd.Series(unit_indexes[:, 0], index=unit_indexes[:, 1], name='Unit')
        
        # join unit column to original data
        asset_units = asset_data.loc[asset_data.index.isin(unit_indexes.keys())].join(unit_indexes).reset_index(drop=True).copy()
        asset_units['Timestep'] = asset_units.groupby('Unit').cumcount() + 1
        segments_data.append(asset_units)
                
        # increase unit count
        unit_count += number+1
        
    segments_data = pd.concat(segments_data)
    segments_data.drop(['label'], axis=1, inplace=True)
    
    # drop null units and re-assign unit label
    null_units = segments_data.loc[(segments_data.iloc[:, 1:].isnull()).any(axis=1), 'Unit']  # skip timestamp - first column
    segments_data = segments_data[~segments_data.Unit.isin(null_units)]
    segments_data.loc[:, 'Unit'] = segments_data.groupby(['Unit']).ngroup().copy() + 1 
    
    segments_data.reset_index(drop=True, inplace=True)
    
    return segments_data


def get_segments_wrapper(DATA_DIR, MIN_DAY_BETWEEN_ERRORS, MAX_DAY_BETWEEN_ERRORS):
    SELECTED_ASSETs = select_assets(DATA_DIR)
    SELECTED_COL = select_columns(SELECTED_ASSETs)
    MERGED_DATA = merge_and_processing(SELECTED_ASSETs, SELECTED_COL)
    SEGMENTS_DATA = get_segments(MERGED_DATA, MIN_DAY_BETWEEN_ERRORS, MAX_DAY_BETWEEN_ERRORS)
    
    return SELECTED_ASSETs, SELECTED_COL, SEGMENTS_DATA