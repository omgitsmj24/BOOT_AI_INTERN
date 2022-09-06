import ast

import pandas as pd
import numpy as np


def read_csv_data(csv_path):
    """Read a csv file data into pandas DataFrame.
    
    Remove a former index column, if it exists. If Timestamp column 
    exists, remove rows with the same Timestamp. Then sort and set 
    Timestamp column into index column of the DataFrame.
    
    Parameters
    ----------
    csv_path : str
        Path to the to loaded csv file
    
    Returns
    -------
    df : pandas.DataFrame
        The loaded DataFrame with Timestamp as index.
    """
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    if 'Timestamp' in df.columns:
        df.drop_duplicates(subset=['Timestamp'], inplace=True)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.sort_values(by='Timestamp', inplace=True)
        df.reset_index(inplace=True, drop=True)

    return df


def encode_set_error_code(df, prefix='error_code_'):
    """Encode set of error code.
    
    Convert set of error code into single columns. Each column 
    represents one error code of the set with values 1 or 0 
    indicating if the error code occurs or not.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to process.
    prefix: str
        The prefix of new columns.
    
    Returns
    -------
    df_copy: pandas.DataFrame
        The DataFrame with new encoded columns.
    """
    df_copy = df.copy()
    error_codes = []
    for set_error in df["set_RfrErrorCode"]:
        set_error = ast.literal_eval(set_error)
        for code in set_error:
            if code not in error_codes:
                error_codes.append(code)
    for code in error_codes:            
        col_name = prefix + str(code)
        values = []
        for set_error in df["set_RfrErrorCode"]:
            set_error = ast.literal_eval(set_error)
            if code in set_error:
                values.append(1)
            else:
                values.append(0)
        df_copy[col_name] = values  

    return df_copy


def remove_rows_invalid_error_code(df, label_column, min_time, max_time):
    """Remove invalid rows.
    
    Error code, that has distance to the previous code not 
    in range [min_time, max_time] will be considered as invalid.
    The rows from a invalid code (inclusive) to previous valid code are
    removed from the DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input data.
    label_column: str
        The column name for the error code.
    min_time : int
        Min time distance between two error codes.
    max_time : int
        Max time distance between two error codes.
        
    Returns
    -------
    pandas.DataFrame
        The new DataFrame removing invalid rows.
    """
    error_code_53 = df.loc[df[label_column]==1]
    error_code_53_indices = error_code_53.index
    nr_error_code_53 = error_code_53_indices.shape[0]
    if nr_error_code_53 <= 1:
        return df
    to_drop_rows_indices = []
    for curr_idx, next_idx in zip(error_code_53_indices[:-1], error_code_53_indices[1:]):
        time_distance = next_idx - curr_idx
        if time_distance < min_time or time_distance > max_time: 
            to_drop_rows_indices += list(range(curr_idx+1, next_idx+1))
    
    to_drop_rows_indices += list(range(error_code_53_indices[0]))
    to_drop_rows_indices += list(range(error_code_53_indices[-1]+1, df.shape[0]))
    df.drop(index=to_drop_rows_indices, inplace=True)
    df.reset_index(inplace=True, drop=True)
    
    return df
