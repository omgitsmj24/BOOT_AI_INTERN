import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def plot_samples(data, plot_cols, plot_units=5, figsize=(10, 3)):
    """
    This function plot values of plot_cols for each unit
    Params:
        data (DataFrame): data to plot
        plot_cols (List): cols to plot
        plot_units (int): numbers of units to plot
        figsize: size of figure
    """

    for col_name in plot_cols:
        # create new figure for each selected column
        plt.figure(figsize=figsize)

        # check maximum units to plot
        plot_units = min(plot_units, len(data['Unit'].unique()))

        for unit in data['Unit'].unique()[:plot_units]:
            unit_data = data[data['Unit'] == unit][col_name]
            plt.plot(unit_data.values.tolist())
        plt.title(col_name)


def plot_neighbors(train_data, val_sample, selected_units, figsize=(10, 3)):
    """
    This function plot values of plot_cols for each unit
    Params:
        data (DataFrame): data to plot
        plot_cols (List): cols to plot
        plot_units (int): numbers of units to plot
        figsize: size of figure
    """

    plt.figure(figsize=figsize)

    for unit in selected_units:
        unit_data = train_data[train_data['Unit'] == unit]['smooth_health_indicator']
        plt.plot(unit_data.values.tolist(), alpha=0.1, c='b')
    plt.plot(val_sample['smooth_health_indicator'].values.tolist(), c='r')
    plt.title('Neighbor plot')


def normalize_data(clustered_data, selected_cols):
    """
    This function apply Standard scaling for clustered data by each cluster
    Params:
        clustered_data (DataFrame): clustered_data
    Returns:
        normalized_data (DataFrame): Normalized data
    """

    # calculate mean and std of each variable for each cluster
    mean_df = pd.DataFrame(columns=selected_cols)
    std_df = pd.DataFrame(columns=selected_cols)

    num_cluster = len(clustered_data.label.unique())
    for label in range(num_cluster):
        cluster_data = clustered_data[clustered_data.label == label]

        std_df = std_df.append(cluster_data[selected_cols].std(), ignore_index=True)
        mean_df = mean_df.append(cluster_data[selected_cols].mean(), ignore_index=True)

    # normalize each cluster's variable
    normalized_data = clustered_data.copy()
    for label in range(num_cluster):
        cluster_data = clustered_data[clustered_data.label == label]
        normalized_data.loc[clustered_data.label == label, selected_cols] = (cluster_data[selected_cols] - mean_df.iloc[
            label]) / (std_df.iloc[label]+ 10**-100)

    # fill nan and inf values with 0
    normalized_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    normalized_data.fillna(0, inplace=True)
    return normalized_data


def cal_health_condition(normalized_data):
    """
    This function apply calculate health condition for each unit. Health condition of an unit at timestep T
    is defined as (max_unit_time - T) / max_unit_time
    Params:
        normalized_data (DataFrame): normalized_data
    Returns:
        normalized_data_with_rul (DataFrame): Normalized data with health condition added to
    """

    normalized_data_with_rul = normalized_data.copy()
    normalized_data_with_rul['health_condition'] = 0

    # get max rul of each unit for scaling
    normalized_data_with_rul['health_condition'] = normalized_data_with_rul.groupby('Unit')['Timestep'].transform(lambda x: (x.max() - x) / x.max())
    return normalized_data_with_rul


def data_fusion(X, weights, units, time, window_size):
    """
    This function calculate health indicator for each unit from sensors data
    Params:
        X (DataFrame): sensors data to run regression
        weights (List): regression model coef
        units (Series): units columns
        time (Series): time columns
        window_size (int): size of smoothing window
    Returns:
        fused_data (DataFrame): data with all sensors combined as health indicator
    """

    # add weight to sensors
    temp = weights * X
    fused_data = pd.DataFrame()

    # construct health indicator ~ sum of weighted values
    fused_data['health_indicator'] = temp.sum(axis=1)

    fused_data['Unit'] = units
    fused_data['Time'] = time

    fused_data['smooth_health_indicator'] = 0

    # smoothing health_indicator with moving mean window
    look_up_size = window_size // 2
    look_back_size = window_size // 2

    for unit in fused_data['Unit'].unique():
        unit_data = fused_data[fused_data.Unit == unit].copy()
        unit_data.reset_index(drop=True, inplace=True)
        unit_data.loc[:, 'smooth_health_indicator'] = unit_data['health_indicator'].rolling(window=window_size,
                                                                                            center=True, min_periods=1).mean()

        fused_data.loc[fused_data.Unit == unit, 'smooth_health_indicator'] = unit_data['smooth_health_indicator'].values

    # offset health_indicator to 1
    for unit in fused_data['Unit'].unique():
        unit_data = fused_data[fused_data.Unit == unit].copy()
        unit_data.reset_index(drop=True, inplace=True)

        # subtract every unit health indicator with its first value
        offset_value = unit_data['smooth_health_indicator'].iloc[0]
        unit_data.loc[:, 'smooth_health_indicator'] = unit_data['smooth_health_indicator'] - offset_value
        fused_data.loc[fused_data.Unit == unit, 'smooth_health_indicator'] = unit_data['smooth_health_indicator'].values

    # start health indicator = 1
    fused_data.loc[:, 'smooth_health_indicator'] = 1 + fused_data['smooth_health_indicator']

    return fused_data


def scoring(train_data, unit_sample, reg_model_lib, sample_len, polynomial_deg=7):
    """
    This function calculate similarity score for each reg_model to 
    query sample
    Params:
        train_data (DataFrame): training data
        unit_sample (DataFrame): query unit to calculate similarity score
        reg_model_lib (dict): linear model of each training unit
        polynomial_deg (int): degree of polynomial line
    Return:
        scores_dict (dict): similarity score of each training unit to query unit
    """

    scores_dict = dict()

    # transform time steps (cycle) to polynomial feature 
    poly = PolynomialFeatures(polynomial_deg)
    time_steps = poly.fit_transform(np.arange(1, sample_len+1).reshape(-1, 1))

    for unit, reg_model in reg_model_lib.items():
        train_unit = train_data[train_data['Unit'] == unit]
        if len(train_unit) < sample_len:
            continue

        # get line of each neighbor
        current_line = reg_model.predict(time_steps)

        # calculate distance between each line and get similarity score
        distance = np.linalg.norm((unit_sample.iloc[:sample_len]['smooth_health_indicator'] - current_line), ord=1)
        sim_score = np.exp(-distance ** 2)

        scores_dict[unit] = sim_score
    return scores_dict


def estimate_rul(selected_units, train_data, sample_len):
    """
    This function estimate RUL of a period given its neighbors RUL info
    Params:
        selected_units (list): neighbor units from training data
        train_data (DataFrame): training data
        sample_len (int): current lifetime value of val sample
    Return:
        predict_RUL (int): estimated RUL 
    """

    # get distribution of next RUL values for each neighbors
    selected_data = train_data[train_data.Unit.isin(selected_units)]
    distributions = selected_data.groupby('Unit').size()
    distributions = pd.Series(distributions)
    
    try:
        ax = distributions.plot(kind='kde', figsize=(10, 5))
        # get distribution values
        hist_x = ax.lines[0]._x
        hist_y = ax.lines[0]._y

        # calculate median of distribution
        cdf = scipy.integrate.cumtrapz(hist_y, hist_x, initial=0)
        nearest_05 = np.abs(cdf - 0.5).argmin()

        x_median = hist_x[nearest_05]
        predict_RUL = max(0, x_median - sample_len)  # predicted value 

    except Exception as e:
        print(e)
        predict_RUL = 0
    plt.close()  # add this so plot wont show up
    return predict_RUL


def predict(train_data, val_fused_data, cutpoint, reg_model_lib, break_point=0, model='linear', NEIGHBORS_NUM=50, polynomial_deg=7):
    """
    This function return loss of a query sample predicted RUL from its neighbors
    Params:
        train_data (DataFrame): train data
        val_fused_data (DataFrame): validation data
        cutpoint (float): percentage of sample length to query
        reg_model_lib (dict): linear model of each training unit
        break_point (int): number of input observations to estimate RUL, this != 0 will ignore cutpoint
        NEIGHBORS_NUM (int): Numbers of neighbor to estimate RUL
        polynomial_deg (int): polynomial degree 
    Return:
        losses (list): loss value list of each query sample
    """

    losses = []
    labels = []
    predict_results = []
    # go through each query unit
    for unit in val_fused_data['Unit'].unique():
        unit_sample = val_fused_data[val_fused_data['Unit'] == unit][['Time', 'smooth_health_indicator']]

        true_RUL_point = int(len(unit_sample) * cutpoint)
        if break_point != 0:
            true_RUL_point = break_point

        true_RUL = len(unit_sample) - true_RUL_point
        
        if model == 'linear':
            scores_dict = scoring_linear_reg(train_data, unit_sample, reg_model_lib, sample_len=true_RUL_point, polynomial_deg=polynomial_deg)
        else:
            scores_dict = scoring(train_data, unit_sample, reg_model_lib, sample_len=true_RUL_point, polynomial_deg=polynomial_deg)

        # sort similarity score asc
        sorted_scores = dict(sorted(scores_dict.items(), key=lambda item: item[1], reverse=True))
        selected_units = list(sorted_scores.keys())[:NEIGHBORS_NUM]  # get top nearest neighbors

        predict_RUL = estimate_rul(selected_units, train_data, true_RUL_point)

        losses.append(abs(predict_RUL - true_RUL))
        labels.append(true_RUL)
        predict_results.append(predict_RUL)

    return losses, labels, predict_results


def consecutive_predict(train_data, val_unit, reg_model_lib, model='linear', polynomial_deg=None, NEIGHBORS_NUM=50):
    """
    This function return prediction RUL of live data
    Params:
        train_data (DataFrame): train data
        val_unit (DataFrame): single-unit validation data
        reg_model_lib (dict): linear model of each training unit
        polynomial_deg (int): polynomial degree of modeled health-indicator
        NEIGHBORS_NUM (int): Numbers of neighbor to estimate RUL
    Return:
        losses (list): loss value list of each query sample
    """
    losses = []
    labels = []
    predict_results = []
    
    min_day_predict = 7  # need at least 7 running days to predict an error
    # go through each query unit
    for i in range(len(val_unit) - min_day_predict):
        sample_len = min_day_predict + i
        true_RUL = len(val_unit) - sample_len
        
        if model == 'linear':
            scores_dict = scoring_linear_reg(train_data, val_unit.iloc[:sample_len], reg_model_lib, sample_len=sample_len, polynomial_deg=polynomial_deg)
        else:
            scores_dict = scoring(train_data, val_unit.iloc[:sample_len], reg_model_lib, sample_len=sample_len, polynomial_deg=polynomial_deg)

        # sort similarity score asc
        sorted_scores = dict(sorted(scores_dict.items(), key=lambda item: item[1]))
        selected_units = list(sorted_scores.keys())[:NEIGHBORS_NUM]  # get nearest neighbors

        predict_RUL = estimate_rul(selected_units, train_data, sample_len)
        predict_results.append(predict_RUL)
        losses.append(abs(predict_RUL - true_RUL))
        labels.append(true_RUL)
        
    return losses, labels, predict_results


def live_predict(train_data, val_fused_data, reg_model_lib, model='linear', polynomial_deg=None, NEIGHBORS_NUM=50):
    """
    This function return prediction RUL of live data
    Params:
        train_data (DataFrame): train data
        val_fused_data (DataFrame): validation data
        reg_model_lib (dict): linear model of each training unit
        polynomial_deg (int): polynomial degree of modeled health-indicator
        NEIGHBORS_NUM (int): Numbers of neighbor to estimate RUL
    Return:
        losses (list): loss value list of each query sample
    """
    predict_results = []
    units = []
    # go through each query unit
    for unit in val_fused_data['Unit'].unique():
        unit_sample = val_fused_data[val_fused_data['Unit'] == unit][['Time', 'smooth_health_indicator']]
        units.append(unit)

        true_RUL_point = len(unit_sample)
        if model == 'linear':
            scores_dict = scoring_linear_reg(train_data, unit_sample, reg_model_lib, sample_len=true_RUL_point, polynomial_deg=polynomial_deg)
        else:
            scores_dict = scoring(train_data, unit_sample, reg_model_lib, sample_len=true_RUL_point, polynomial_deg=polynomial_deg)

        # sort similarity score asc
        sorted_scores = dict(sorted(scores_dict.items(), key=lambda item: item[1]))
        selected_units = list(sorted_scores.keys())[:NEIGHBORS_NUM]  # get nearest neighbors

        predict_RUL = estimate_rul(selected_units, train_data, true_RUL_point)
        predict_results.append(predict_RUL)

    return predict_results, units


def fit_polyline(train_data, polynomial_deg):
    """
    This function return fitted polynomial line for each training data period
    Params:
        train_data (DataFrame): train data with smoothed health indicator value
        polynomial_deg (int): degree of polynominal curve
    Return:
        reg_model_lib (dict): fit line for each unit in train data
    """
    
    # fit polynominal line to health-indicator lines
    reg_model_lib = dict()
    vis_loss = []
    for unit in train_data['Unit'].unique():
        # get each unit data
        unit_data = train_data[train_data['Unit'] == unit]
        unit_data.reset_index(drop=True, inplace=True)

        # convert to polynomial feature 
        X_poly = unit_data.Time
        y_poly = unit_data['smooth_health_indicator']

        poly = PolynomialFeatures(degree=polynomial_deg)
        X_poly = poly.fit_transform(X_poly.values.reshape(-1, 1))

        # fit poly line
        clf = LinearRegression()
        clf.fit(X_poly, y_poly)

        # save unit line to query dict
        reg_model_lib[unit] = clf
    return reg_model_lib


def scoring_linear_reg(train_data, unit_sample, reg_model_lib, sample_len, polynomial_deg=7):
    """
    This function calculate similarity score for each reg_model to 
    query sample
    Params:
        train_data (DataFrame): training data
        unit_sample (DataFrame): query unit to calculate similarity score
        reg_model_lib (dict): linear model of each training unit
        polynomial_deg (int): degree of polynomial line
    Return:
        scores_dict (dict): similarity score of each training unit to query unit
    """

    scores_dict = dict()

    # transform time steps (cycle) to polynomial feature 
    poly = PolynomialFeatures(polynomial_deg)
    time_steps = poly.fit_transform(np.arange(1, sample_len+1).reshape(-1, 1))
    time_steps = pd.DataFrame(time_steps)
    
    # select units with valid length only
    train_unit_sizes = train_data.groupby('Unit').size()
    pred_units = train_unit_sizes[train_unit_sizes >= sample_len].index.values

    # convert dict of reg models to dataframe
    model_params = []
    for unit in pred_units:
        reg_model = reg_model_lib[unit]
        model_params.append(reg_model.coef_.tolist() + [reg_model.intercept_]+[unit])
    model_params = pd.DataFrame(model_params)
    
    # renaming 
    model_params.rename(columns={model_params.columns[-1]: "Unit",
                                 model_params.columns[-2]: "intercept"},
                        inplace = True)
    
    # calculate distance and similarity score
    neighbors_line = time_steps.dot(model_params.iloc[:, :-2].T) + model_params['intercept']
    distance = unit_sample.iloc[:sample_len]['smooth_health_indicator'].values - neighbors_line.T
    distance['Unit'] = model_params['Unit'].copy()
    distance['sim_score'] = np.exp(-np.linalg.norm(distance.T.iloc[:-1, :], ord=1, axis=0) ** 2)
    
    scores_dict = distance[['Unit', 'sim_score']].set_index('Unit')['sim_score'].to_dict()
    return scores_dict


def get_loss(train_data, val_fused_data, reg_model_lib, polynomial_deg, NEIGHBORS_NUM):
    """
    This function return experimental loss of current modeled health-indicator line.
    The result consists of different losses from 3 input sequence lengths - 50%, 70% and 90% length of input
    Params:
        train_data (DataFrame): train data
        val_fused_data (DataFrame): validation data
        reg_model_lib (dict): linear model of each training unit
        polynomial_deg (int): polynomial degree of modeled health-indicator
    Return:
        loss_df (DataFrame): loss value list of each query sample
    """
    
    # get losses
    loss_50, labels_50, pred_50 = predict(train_data, val_fused_data, 0.5, reg_model_lib, model='linear', break_point=0, NEIGHBORS_NUM=NEIGHBORS_NUM, polynomial_deg=polynomial_deg)
    loss_70, labels_70, pred_70 = predict(train_data, val_fused_data, 0.7, reg_model_lib, model='linear', break_point=0, NEIGHBORS_NUM=NEIGHBORS_NUM, polynomial_deg=polynomial_deg)
    loss_90, labels_90, pred_90 = predict(train_data, val_fused_data, 0.9, reg_model_lib, model='linear', break_point=0, NEIGHBORS_NUM=NEIGHBORS_NUM, polynomial_deg=polynomial_deg)

    loss_df = pd.DataFrame()
    loss_df['True RUL 50'] = labels_50
    loss_df['Pred RUL 50'] = pred_50
    loss_df['Sample 50%'] = loss_50

    loss_df['True RUL 70'] = labels_70
    loss_df['Pred RUL 70'] = pred_70
    loss_df['Sample 70%'] = loss_70

    loss_df['True RUL 90'] = labels_90
    loss_df['Pred RUL 90'] = pred_90
    loss_df['Sample 90%'] = loss_90
    return loss_df

def SlopeRanker(data, data_variables, nsample):
    
    """ 
    This function ranks the important of sensors by measuring the slope of each 
    sensor data output signal and rank them with the highest being the most important
    
    Params:
        data (DataFrame): Input scaled data
        data_variables (DataFrame): specific sensors to calculate slope and rank
        nsample (int): number of machine to calculate slope
    Return: 
        Final_Results (DataFrame): a DataFrame with slope calculated and sorted from highest to lowest
    """
    
    numSensors = len(data_variables)
    signalSlope = np.zeros(numSensors)
    Final_Result = np.zeros(numSensors)
    for ct in range (numSensors):
        sum = []
        for i in range (1, 1 + nsample): 
            lr = LinearRegression()
            y = data[data.Unit == i][data_variables[ct]]
            y = y.values
            y = y.reshape(-1,1)
            X = data[data.Unit == i].Timestep
            X = X.values
            X = X.reshape(-1,1)
            lr.fit(X,y)
            signalSlope[ct] += abs(lr.coef_)  
            Final_Result[ct] = signalSlope[ct] / nsample
            # print(f'Sensor {ct+1} in machine {i} has slope: {signalSlope[ct]}')
        # print('-')
        # print(f'Sensor {ct+1} has slope: {Final_Result[ct]}')
        # print('----------------------------------------')
    
    # signalSlope.sort()
    return Final_Result

    