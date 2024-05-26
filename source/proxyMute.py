import collections
import copy
from random import random

import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt

from bias_measures import test_bias
from data_loader import get_features, get_gender
from training import blackbox_regressor

DATA_PATH = "../faircvtest/"


def set_disabling_rules():
    dict_ = {'education': 'median', 'recommendation': 'median',
             'availability': 'median', 'prev_exp': 'median', 'lang': 'median', 'face': 'mean'}
    return dict_


def nullfy_given_indices(df, test_df, index_arr, strategy='mean'):
    dict_ = set_disabling_rules()

    df_new = copy.deepcopy(df)
    test_df_new = copy.deepcopy(test_df)
    if strategy == 'median':
        mean_df = df_new.median().T.to_frame()
    elif strategy == 'category':
        mean_df = df_new.mode().T.to_frame()
    else:
        mean_df = df_new.mean().T.to_frame()

    mean_df['feat_name'] = mean_df.index
    mean_df = mean_df.reset_index(drop=True)
    for data in [df_new, test_df_new]:
        for index in index_arr:
            feature = mean_df.loc[index]['feat_name']
            is_set = False
            if strategy == 'constant':
                data[feature] = 0
            else:
                for val in dict_.keys():
                    if val in feature:
                        if dict_[val] == 'median':
                            data[feature] = df_new.median().T.to_frame()[0][feature]
                            is_set = True
                if not is_set:
                    data[feature] = mean_df.loc[index][0]
    return df_new, test_df_new


def get_expl(filename, model, data):

    shap_values = shap.TreeExplainer(model).shap_values(get_features(data['test']))
    important_feat = pd.DataFrame(abs(shap_values).mean(0)).reset_index(drop=True)

    important_feat.to_csv("../faircvtest/sex_relevant_feat_{file}.csv".format(file=filename))


def ours_(fold, folder, method=''):
    # choose the model that gives the lowest score for lambda * mae on validation set.

    df = collections.defaultdict(list)
    for test_type in ['biased']:
        for split in ['test']:
            read = test_type
            df[split] = pd.read_csv(
                folder + "bias_iterative_analysis_" + read + "-" + split  + f"_{fold}.csv")
            df[split] = df[split][(df[split]['exp'] == 'shap-gender') & (df[split]['method'] == 'feat-disabling')
                                  & (df[split]['i'] != 0)]

            df[split]['ID'] = df[split].index
        func_methods = ['FUNC_' + str(lambda_val) for lambda_val in [0, 0.25, 0.4, 0.5, 0.6, 0.7, 0.75, 1]]
        dict_m = collections.defaultdict(list)
        df['val'] = df['test']
        for m in ['EA', 'PCC', 'MEAN'] + func_methods:
            metric = m
            if m in ['EA', 'PCC']:
                df['val'] = df['val'].sort_values(by=[m], ascending=True)

            elif m == 'MEAN':
                df['val']['fairness_score'] = [(abs(row['PCC']) + abs(row['EA'])) / 2 for index, row in
                                               df['val'].iterrows()]
                df['val'] = df['val'].sort_values(by=['fairness_score'], ascending=True)
                metric = 'fairness_score'

            if m.startswith('FUNC_'):
                lambda_val = float(m.split('_')[1])
                df['val']['fairness_score'] = [(abs(row['PCC']) + abs(row['EA'])) / 2 for index, row in
                                               df['val'].iterrows()]

                df['val']['threshold_func'] = [
                    (lambda_val * (float(row["fairness_score"]))) + ((1 - lambda_val) * (float(row["MAE"]))) for
                    index, row in df['val'].iterrows()]
                df['val'] = df['val'].sort_values(by=['threshold_func'], ascending=True)
                metric = 'threshold_func'

            index = \
            df['val'][df['val'][metric] == df['val'].iloc[0][metric]].sort_values(by=['ID'], ascending=True).iloc[0][
                'ID']

            df['test']['ID'] = df['test'].index
            ea, mae, pcc, id, method_, exp_ = \
                df['test'][df['test']['ID'] == index][["EA", "MAE", "PCC", 'i', 'method', 'exp']].values[0]
            dict_m[m] = [ea, mae, pcc, id, method_, exp_]

        df_all = pd.DataFrame.from_dict(dict_m)
        df_all.index = ['EA', 'MAE', 'PCC', 'i', 'method', 'exp']
        df_all.to_csv(folder + "ours_{type}_{method}.csv".format(type=test_type, method=method))
    return dict_m


def iterative_analysis_(file, profiles, train_labels, test_labels, final_file_name):
    attr = 'gender'
    arr_sub = collections.defaultdict(list)

    print(f"READING FILE ../faircvtest/sex_relevant_feat_{file}.csv")
    df = pd.read_csv("../faircvtest/sex_relevant_feat_{file}.csv".format(file=file))
    df['feat'] = get_features(profiles['test']).columns
    df['index'] = df.index
    df = df.sort_values(by=['0'], ascending=False)
    for del_, arr in [["shap-gender", df['index']]]:
        for method in ['feat-disabling']:
            test_preds, regressor = blackbox_regressor(get_features(profiles['train']), get_features(profiles['test']),
                                                       train_labels, test_labels)

            for i in range(int(len(arr) - 1)):
                del_val = list(arr[0:i])
                if method == 'feat-disabling':
                    train_data, test_data = nullfy_given_indices(get_features(profiles['train']),
                                                                 get_features(profiles['test']), del_val)
                    test_preds = regressor.predict(test_data)
                elif method == 'ROAR':
                    train_data = remove_given_indices(get_features(profiles['train']), del_val)
                    test_data = remove_given_indices(get_features(profiles['test']), del_val)
                    test_preds, regressor = blackbox_regressor(train_data,
                                                               test_data,
                                                               train_labels,
                                                               test_labels)

                PCC, EAD, MAE = test_bias(get_gender(profiles['test']), test_preds, test_labels)
                for str, val in [['EA', EAD], ['MAE', MAE], ['PCC', PCC], ['i', i], ['method', method], ['exp', del_]]:
                    arr_sub[str].append(val)

    df = pd.DataFrame().from_dict(arr_sub, orient='index').transpose()
    df.to_csv(f"../faircvtest/bias_iterative_analysis_{final_file_name}.csv".format(final_file_name=final_file_name))
    print(f"SAVING ITERATIVE ANALYSIS FILE TO ../faircvtest/bias_iterative_analysis_{final_file_name}.csv")
    return


def remove_given_indices(df, index_arr):
    df_new = df.T.reset_index(drop=True)
    df_new = (df_new.loc[~df_new.index.isin(index_arr), :])
    return (df_new.T)
