import os
import ast
import random
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from glob import glob as gl
import matplotlib.pyplot as plt

MISSING_DAY_LABEL = -100
FREQUENT_ERROR_LABEL = -101


def read_data(path, error_code_label="error_code_53"):
    """
    Function used to read dataframe by asset_id from csv file
    """
    asset_message = pd.read_csv(path)
    asset_message = asset_message.loc[:, ~asset_message.columns.str.contains('^Unnamed')]
    asset_message["Timestamp"] = asset_message["Timestamp"].astype('datetime64[ns]')
    asset_message["Timestamp"] = asset_message["Timestamp"].dt.date
    asset_message['Timestamp'] = pd.to_datetime(asset_message['Timestamp'])
    
    # get unique error codes
    # for larger file use this
    # error_codes = np.unique(np.concatenate(asset_message['set_RfrErrorCode'].apply(ast.literal_eval)))
    error_codes = [53, 91, 0]
    
#     for set_error in asset_message["set_RfrErrorCode"]:
#         set_error = ast.literal_eval(set_error)
#         for code in set_error:
#             if code not in error_codes:
#                 error_codes.append(code)
    
    # create new error columns
    for code in error_codes:
        col_name = "error_code_" + str(code)
        values = asset_message["set_RfrErrorCode"].str.contains(str(code)).values.astype(int)
        asset_message.loc[:, col_name] = values

#     asset_message.rename(columns={error_code_label: "label"}, inplace=True)
#     try:
#         asset_message.label = asset_message.label.astype(int)
#     except:
#         asset_message['label'] = 0
    return asset_message

def pre_consecutive_label(labels, error_label=1, window_len=3):
    """
    Function used to label pre-consecutive day from 53 error code day (previous days)
    Params:
        labels: (list) label column value
        error_label: (int) label of error code 53
        window_len: (int) numbers of previous days to label
    Return: New list labels
    """

    new_labels = np.array(labels)
    keep_label = [error_label, MISSING_DAY_LABEL, FREQUENT_ERROR_LABEL]

    for i in range(len(new_labels) - 1, 0, -1):
        if labels[i] == error_label:  # if current day has error, sequentially label D previous day
            for j in range(window_len):
                assign_index = i - j - 1
                valid_index = assign_index >= 0
                if valid_index and labels[assign_index] not in keep_label:
                    new_labels[assign_index] = -j - 1
    return new_labels.astype(int)


def remove_frequent_errorcode(group, new_label=-101, error_label=1, window_len=7):
    """
    Function used to remove frequent error code 53 in the next D-days
    Params:
        group: (list) label column value
        new_label: (int) new label assigned to missing day
        error_label: (int) label of error code 53
        window_len: (int) amount days to check
    Return: new list labels
    """

    return_group = np.array(group)
    index = 0
    while index < len(return_group):
        if group[index] == error_label:  # if current day has error, check next D day if error occurs again
            for j in range(window_len):
                assign_index = index + j + 1
                valid_index = assign_index < len(return_group)
                if valid_index and group[assign_index] == error_label:
                    return_group[index + 1: assign_index + 1] = new_label  # assign new label if another error occurs
                    break
            index = assign_index  # start new index from found error code
        else:
            index += 1
    return return_group


def fill_missing_day(data):
    """
    Function to fill missing day value
    Params:
        data: dataframe
    Return: datafame
    """
    idx = pd.date_range(data['Timestamp'].min(), data['Timestamp'].max())
    new_data = pd.DataFrame(idx)
    new_data.rename(columns={0: "Timestamp"}, inplace=True)
    data = data.merge(new_data, on='Timestamp', how='right')
    data.reset_index(inplace=True, drop=True)
    return data


def labeling(data, window_freq, window_label):
    """
    Function used to labeling record
    Params:
        data: dataframe
    Return: datafame
    """
    data['label'] = data['label'].fillna(-100)  # missing day label is -100
    data['frequent_error_label'] = remove_frequent_errorcode(data['label'], new_label=-101,
                                                             error_label=1, window_len=window_freq)
    data['consecutive_label'] = pre_consecutive_label(data['frequent_error_label'],
                                                      error_label=1, window_len=window_label)
    return data


def remove_high_corr_columns(data, threshold):
    """
    Function used to remove columns which have high correlation value
    Params:
        data: dataframe
        threshold: float, 0->1
    Return: dataframe after drop columns
    """

    error_col = []
    for col in data.columns:
        if 'error_code' in col:
            error_col.append(col)
    keep_col = error_col + ['label', 'consecutive_label', 'frequent_error_label', 'set_AssetId']
    label_data = data[keep_col]

    corr_matrix_abs = data.drop([keep_col], axis=1).corr().abs()
    upper = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    float_data_remove_high_corr_features = data.drop(columns=to_drop)

    float_data_remove_high_corr_features.merge(label_data, how='inner')
    return float_data_remove_high_corr_features


def visualize_distribution(vis_df, asset_id):
    """
    Function use to visualize each feature distribution with kde and save plots
    Params:
        vis_df: dataframe to visualize
        asset_id: asset id - use as saving directory for all plots
    :return:
    """
    plot_colors = dict()
    selected_color = ['red', 'green', 'blue']  # D - 0 - 1
    for index, label in enumerate(sorted(vis_df.consecutive_label.unique())):
        plot_colors[str(label)] = selected_color[index]

    vis_df.consecutive_label = vis_df.consecutive_label.astype(int)
    features = vis_df.columns.to_list()
    features.remove('consecutive_label')
    features.remove('label')
    features.remove('set_AssetId')
    query_class = "consecutive_label"
    n_features = len(vis_df.columns) - 2
    for i in range(n_features):
        query_feature = features[i]
        feature_id = query_feature.split('_')[1]  # get feature name

        # create save dir
        dir_name = os.path.join('results', asset_id, feature_id)
        save_path = os.path.join(dir_name, f'{query_feature}_distribution.png')
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # plots
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), facecolor='w', edgecolor='k')
        fig.suptitle(asset_id + '\n' + query_feature, fontsize=15)
        axs = axs.ravel()
        vis_df.groupby(query_class)[query_feature].hist(alpha=0.4, ax=axs[0])
        try:
            sns.kdeplot(data=vis_df, x=query_feature, hue=query_class, color=plot_colors)
        except Exception as e:
            print(query_feature, e)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def statistic(statistic_data, visual_day, sort_by='pvalue_ttest'):
    """
    Function used to calculate pvalus of ttest statistic of all columns
    
    :param data: dataframe
    :param data: pvalue_ttest/pvalue_wilcoxon
    :return: pvalue
    """
    data = statistic_data.copy()
    pvalue_result = []
    # Get all number columns except label
    except_columns = ['frequent_error_label', 'consecutive_label', 'label', 'set_AssetId']
    for col in data.columns:
        if 'error' in col:
            except_columns.append(col)
    all_columns = data.drop(except_columns, axis=1).columns.values.tolist()
    for column in all_columns:
        # Get previous data of error53 day
        normal_data = data.loc[data['consecutive_label'] == 0][column].values.tolist()
        pre_error53_data = data.loc[(data['consecutive_label'] == -visual_day)][column].values.tolist()

        # Statistic t test
        ttest_result = stats.ttest_ind(pre_error53_data, normal_data)

        # Wilcoxon rank sum test
        wil_result = stats.ranksums(pre_error53_data, normal_data)

        # Save result
        pvalue_result.append([column, ttest_result.pvalue, wil_result.pvalue])
    pvalue_df = pd.DataFrame(pvalue_result, columns=['feature', 'pvalue_ttest', 'pvalue_wilcoxon'])
    pvalue_df.sort_values(by=sort_by, na_position='last', inplace=True)

    return pvalue_df


def fill_missing_data(data, amount_day_avg=2):
    """
    Function used to fill missing value of all float columns by average by 'D-days' (front or back).
    
    :param data: dataframe
    :param amount_day_avg: amount days
    :return: dataframe after fill missing valuese
    """
    # data: float_data_except_error code and labels
    for column in data.columns:
        if 'stdev_' in column:
            # Fill 0 value (top -> down)
            for i in range(amount_day_avg, len(data[column]), 1):
                if data[column][i] == 0:
                    # fill if min value on that day is nan
                    if (f'max_{column[6:]}' in list(data.columns) and np.isnan(data[f'max_{column[6:]}'][i])) or \
                            (f'min_{column[6:]}' in list(data.columns) and np.isnan(data[f'min_{column[6:]}'][i])):
                        data[column][i] = np.average(data[column][i - amount_day_avg:i])

            # Fill nan value (down -> top)            
            for i in range(len(data[column]) - amount_day_avg - 1, -1, -1):
                if data[column][i] == 0:
                    # fill if min value on that day is nan
                    if (f'max_{column[6:]}' in list(data.columns) and np.isnan(data[f'max_{column[6:]}'][i])) or \
                            (f'min_{column[6:]}' in list(data.columns) and np.isnan(data[f'min_{column[6:]}'][i])):
                        data[column][i] = np.average(data[column][i + 1:i + amount_day_avg + 1])
        else:
            # Fill nan top -> down
            for i in range(amount_day_avg, len(data[column]), 1):
                if np.isnan(data[column][i]) and np.nan not in data[column][i - amount_day_avg:i]:
                    data[column][i] = np.average(data[column][i - amount_day_avg:i])
                    # Fill nan down -> top
            for i in range(len(data[column]) - amount_day_avg - 1, -1, -1):
                if np.isnan(data[column][i]) and np.nan not in data[column][i + 1:i + amount_day_avg + 1]:
                    data[column][i] = np.average(data[column][i + 1:i + amount_day_avg + 1])
    return data


def plot_one_feature(asset_ids, column_name, step=1):
    """
    Function used to plot one unit of some asset_id (timeseries plot)
    
    :param asset_ids: list asset_id
    :param column_name: column name
    :param step: ordinal of unit
    :return: plot
    """
    column_value = []
    columns = []
    for asset_id in asset_ids:
        data = pd.read_csv(f'useful_data/{asset_id}/not_have_label.csv')
        data = data.rename(columns={'Unnamed: 0': 'no'})
        data = data.loc[data['step'] == step]
        if column_name in data.columns:
            column_value.append(data[column_name].values.tolist())
            columns.append(asset_id.split('-')[-2])
    df = pd.DataFrame(column_value, dtype=float).T
    df.columns = columns
    df.plot(figsize=(20, 8), title=column_name, ylabel='sensor value')
    plt.show()


def compute_distance_between_error_code(df, error_column,
                                        min_time_between_errors, max_time_between_errors):
    """Compute time distance between two error code.
    
    Return a DataFrame containing time distance unique values
    and corresponding number of occurrences. Filter time distance
    that not in range [min_time_between_errors, max_time_between_errors].
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing error code for counting.
    error_column : str
        Name of error code column in the DataFrame.
    min_time_between_errors : int
        The minimum value for time distance.
    max_time_between_errors : int
        The maximum value for time distance.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame with two columns: unique values for time distances
        and number of occurrences.
    """
    error_code = df.loc[df[error_column] == 1]
    distance_between_error_code = error_code.index[1:] - error_code.index[:-1]
    distance_between_error_code = distance_between_error_code.to_frame()
    count_df = distance_between_error_code.value_counts()
    count_df.sort_index(inplace=True)
    count_df = count_df.loc[min_time_between_errors: max_time_between_errors]
    count_df = count_df.rename_axis('time_distance').reset_index(name='count')

    return count_df
