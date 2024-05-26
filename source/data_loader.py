import collections

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

DATA_PATH = "../faircvtest/"

def split_test_data():
    dict_profiles = np.load(f'{DATA_PATH}/Profiles_test_gen.npy', allow_pickle=True).item()
    profiles_feat = (dict_profiles['profiles'])
    biased_labels = dict_profiles['biasedLabels']
    blind_labels = dict_profiles['blindLabels']

    male_indices = np.where(profiles_feat[:, 1] == 0)
    female_indices = np.where(profiles_feat[:, 1] == 1)
    male_test = np.random.choice(male_indices[0], size=1200, replace=False)
    female_test = np.random.choice(female_indices[0], size=1200, replace=False)

    index = np.concatenate([male_test, female_test])
    df = pd.DataFrame(profiles_feat)
    df1 =  df[df.index.isin(index)].to_numpy()
    df2 = df[~df.index.isin(index)].to_numpy()
    blind_labels = pd.DataFrame(blind_labels)
    biased_labels = pd.DataFrame(biased_labels)

    val_dict = collections.defaultdict(list)
    val_dict['profiles'] = df1
    val_dict['biasedLabels'] = biased_labels[biased_labels.index.isin(index)].to_numpy()
    val_dict['blindLabels'] = blind_labels[blind_labels.index.isin(index)].to_numpy()

    test_dict = collections.defaultdict(list)
    test_dict['profiles'] = df2
    test_dict['biasedLabels'] = biased_labels[~biased_labels.index.isin(index)].to_numpy()
    test_dict['blindLabels'] =  blind_labels[~blind_labels.index.isin(index)].to_numpy()

    np.save(f"{DATA_PATH}/Profiles_new_test_gen.npy", test_dict)
    np.save(f"{DATA_PATH}/Profiles_val_gen.npy", val_dict)


def read_data(split="train"):
    file_ = split
    dict_profiles = np.load(DATA_PATH + "Profiles_" + file_ + "_gen.npy", allow_pickle=True).item()
    profiles_feat = (dict_profiles['profiles'])
    biased_labels = dict_profiles['biasedLabels']
    blind_labels = dict_profiles['blindLabels']

    return profiles_feat, blind_labels, biased_labels


def get_eth(profiles_feat):
    return profiles_feat[:, 0]  # 0 = G1, 1 = G2, 3 = G3


def get_gender(profiles_feat):
    return profiles_feat[:, 1] # 0 = Male, 1 = Female


def get_educ_attainment(profiles_feat):
    return profiles_feat[:, 2]  # Discrete variable [0 - 5]


def get_prev_experience(profiles_feat):
    return profiles_feat[:, 3]  # Continuous variable [0 - 4]


def get_recommendation(profiles_feat):
    return profiles_feat[:, 4]  # Binary variable


def get_availability(profiles_feat):
    return profiles_feat[:, 5]  # Discrete variable [1 - 5]


def get_language_prof(profiles_feat):
    return profiles_feat[:, 6:14]  # Discrete variables [0 - 3]


def get_face_embedding(profiles_feat, del_arr=False):
    embedding = profiles_feat[:, 14:34]
    if del_arr == True:
        embedding = np.delete(embedding, del_arr, axis=1)
    return pd.DataFrame(embedding)


def get_agnostic_face_embedding(profiles_feat):
    return pd.DataFrame(profiles_feat[:, 34:])


def get_categorical_feat(profiles_feat):
    df = pd.DataFrame()
    df['education'] = get_educ_attainment(profiles_feat)
    df['recommendation'] = get_recommendation(profiles_feat)
    df['availability'] = get_availability(profiles_feat)
    df['prev_exp'] = get_prev_experience(profiles_feat)
    df_lang = pd.DataFrame(get_language_prof(profiles_feat))
    df_lang.columns = ['lang_' + str(col) for col in df_lang.columns]
    df = pd.concat([df, df_lang], axis=1)
    return df

def get_features(profiles_feat, agnostic=False, del_arr=False):
    if agnostic:
        df = get_agnostic_face_embedding(profiles_feat)
    else:
        df = get_face_embedding(profiles_feat)
    df.columns = ['face_' + str(col) for col in df.columns]
    df['education'] = get_educ_attainment(profiles_feat)
    df['recommendation'] = get_recommendation(profiles_feat)
    df['availability'] = get_availability(profiles_feat)
    df['prev_exp'] = get_prev_experience(profiles_feat)

    df_lang = pd.DataFrame(get_language_prof(profiles_feat))
    df_lang.columns = ['lang_' + str(col) for col in df_lang.columns]
    df = pd.concat([df, df_lang], axis=1)
    if ((del_arr) and (del_arr != None)):
        df = pd.DataFrame(np.delete(df.to_numpy(), del_arr, axis=1))
    return df.reset_index(drop=True)

# val_profiles, val_blind_labels, val_biased_labels = read_data("val")

def load_cv():
    train_profiles, train_blind_labels, train_biased_labels = read_data("train")
    test_profiles, test_blind_labels, test_biased_labels = read_data("test")

    cv_profiles = np.concatenate((train_profiles, test_profiles), axis=0)
    cv_biased_labels = np.concatenate((train_biased_labels, test_biased_labels), axis=0)
    cv_blind_labels = np.concatenate((train_blind_labels, test_blind_labels), axis=0)

    skf = StratifiedKFold(n_splits=10)

    data = collections.defaultdict(list)
    for i, (train_index, test_index) in enumerate(skf.split(cv_profiles, get_gender(cv_profiles))):
        data['fold_'+str(i)] = collections.defaultdict(list)
        data['fold_' + str(i)]['profiles'] = collections.defaultdict(list)
        data['fold_' + str(i)]['biased_labels'] = collections.defaultdict(list)
        data['fold_' + str(i)]['blind_labels'] = collections.defaultdict(list)

        for split, index in [['train', train_index], ['test', test_index]]:
            data['fold_'+str(i)]['profiles'][split] = cv_profiles[index]
            data['fold_'+str(i)]['biased_labels'][split] = cv_biased_labels[index]
            data['fold_'+str(i)]['blind_labels'][split] = cv_blind_labels[index]

    return data

data = load_cv()

if __name__ == "__main__":
   print(data.keys())