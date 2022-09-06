import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.impute import KNNImputer
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm import tqdm

from utils import preprocessing


def decompose_time_series(ts, col_name, period, error_idx, save_path):
    """Decompose the given time series.
    
    The time series is decomposed in three components trend, seasonal and
    residual. The observed data with the components are plotted and saved
    in folder `save_path`. The time, when the error code occurs will be 
    marked with a vertical line.
    
    Parameters
    ----------
    ts : array-like
        Time series data.
    col_name : str
        Sensor name.
    period : int
        The period for time series decomposition.
    error_idx : list
        A list of index of error codes.
    save_path : str
        The path, where the plots will be saved.
    """
    sns.set(style="darkgrid")
    plt.rc('figure', figsize=(20, 9))
    
    decomposition = seasonal_decompose(ts, model='additive', period=period)
    fig, axes = plt.subplots(4, 1, sharex=True)
    
    decomposition.observed.plot(ax=axes[0], legend=False, color='r')
    axes[0].set_ylabel('Observed')
    decomposition.trend.plot(ax=axes[1], legend=False, color='g')
    axes[1].set_ylabel('Trend')
    decomposition.seasonal.plot(ax=axes[2], legend=False, color='b')
    axes[2].set_ylabel('Seasonal')
    decomposition.resid.plot(ax=axes[3], legend=False, color='purple')
    axes[3].set_ylabel('Residual')
    
    # Plot vertical lines when error code occurs
    for i in range(4):
        for idx in error_idx:
            axes[i].axvline(x=idx, ymin=0, ymax=1, color='navy')
    plt.suptitle(col_name, fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    
    
def visualize_decomposition_time_series(csv_path, label_column, save_dir, min_time_between_2_error, 
                                        max_time_between_2_error, nan_threshold):
    """Visualize decomposition results for sensor.
    
    Each sensor time series are decomposed in three components the trend,
    seasonal and residual. Error code, that has distance to the previous code not 
    in range [min_time, max_time] will be considered as invalid.
    The rows from a invalid code (inclusive) to previous valid code are
    removed from the DataFrame. Features with nan percentage greater than threshold
    are not considered. The plots will be saved in `save_dir` folder.
    
    Parameters
    ----------
    csv_path : str
        The path to data csv file.
    label_column : str
        The column name of the error code label containing 0 and 1.
    save_dir : str
        The folder for saving plots.
    min_time_between_2_error : int
        The minimum time between two error codes.
    max_time_between_2_error : int
        The maximum time between two error codes.
    nan_threshold : float
        The threshold for nan value percentage.
    """
 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df = preprocessing.read_csv_data(csv_path)
    df = preprocessing.encode_set_error_code(df)
    df = preprocessing.remove_rows_invalid_error_code(df, label_column, min_time_between_2_error, max_time_between_2_error)
    # Get float colums
    df = df.loc[:, ((df.dtypes==float) | (df.dtypes==np.int64))]
    # Remove column with nan values percentage greater than nan_threshold
    df = df.loc[:, df.isnull().mean() < nan_threshold]
    # Fill nan values uign KNNImputer
    df_cols = df.columns
    for col in df_cols:
        imputer = KNNImputer(n_neighbors=10, weights="uniform")
        df[col] = imputer.fit_transform(df[col].values.reshape(-1, 1))
    
    label = df[label_column]
    nr_cycles = label[label==1].shape[0] - 1
    error_indices = list(label.loc[label==1].index)
    for col in tqdm(df_cols):
        if col == label_column:
            continue
        save_path = os.path.join(save_dir, col + '.png')
        decompose_time_series(df[col], col, nr_cycles, error_indices, save_path)
