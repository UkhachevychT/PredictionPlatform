import copy

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def test(X, Y, model):
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        return accuracy


def correlation(dataframe, threshold):
    df = copy.deepcopy(dataframe)
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df.drop(to_drop, axis=1, inplace=True)

    return df


def apply_pca(data, data_to_predict=None):
    pca = PCA(.90)
    proccessed_data = pca.fit(data).transform(data_to_predict if data_to_predict is not None else data)
    print(f'PCA: {data.shape[1]} to {proccessed_data.shape[1]}')

    return proccessed_data


def apply_lda(data_X, data_Y, x_to_predict=None):
    lda = LinearDiscriminantAnalysis(n_components=1)
    proccessed_data = lda.fit(data_X, data_Y).transform(x_to_predict if x_to_predict is not None else data_X)
    print(f'LDA: {data_X.shape[1]} to {proccessed_data.shape[1]}')

    return proccessed_data


def apply_var_thres(data, threshold=0.5):
    try:
        selector = VarianceThreshold(threshold)
        proccessed_data = selector.fit_transform(data)
        print(f'var_thres: {data.shape[1]} to {proccessed_data.shape[1]}')

        return np.array(proccessed_data)
    except Exception as err:
        print(f"ERROR! {err}")
        return np.array(data)


def apply_cor_thres(data, threshold=0.5):
    proccessed_data = correlation(data, threshold)
    print(f'cor_thres: {data.shape[1]} to {proccessed_data.shape[1]}')

    return np.array(proccessed_data)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def scale_data(X, method=None, fit_on=None):
    X_scaled = None
    if method == 'z':
        scaler_z = StandardScaler()
        X_scaled = pd.DataFrame(scaler_z.fit(fit_on if fit_on is not None else X).transform(X))
        # X_scaled = pd.DataFrame(scaler_z.fit_transform(X))
    else:  # minmax
        scaler_min_max = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler_min_max.fit(fit_on if fit_on is not None else X).transform(X))
        # X_scaled = pd.DataFrame(scaler_min_max.fit_transform(X))
    return X_scaled


def reduc_data(x, y=None, method=None, x_to_predict=None):
    reduc_X = object()

    if method == 'pca':
        reduc_X = apply_pca(x, x_to_predict)
    # elif method=='var_thres':
    #     reduc_X = apply_var_thres(x)
    # elif method=='cor_thres':
    #     reduc_X = apply_cor_thres(x)
    else:  # lda
        reduc_X = apply_lda(x, y, x_to_predict=x_to_predict)
    return reduc_X


def choose_procesing_methods(dataframe):
    reduc_func = 'pca' if len(dataframe) > 500 else 'lda'
    norm_func = 'minmax'
    for x, y in dataframe.iteritems():
        if max(y)-y.quantile(.75) > 3*y.std() or \
                y.quantile(.25)-min(y) > 3*y.std():
            norm_func = 'z'
            break
    return reduc_func, norm_func


def process_data(data_file, categorical_values=None, target='target', is_training_data=True, initial_data=None,
                 norm_func=None, reduc_func=None):
    if is_training_data:
        if type(data_file) == dict:
            df = pd.DataFrame(data_file.items())
        else:
            df = pd.read_csv(data_file)
        df_copy = copy.deepcopy(df)
        x_header_names = list(df.drop([target], 1))

        if df.isnull().sum().sum() != 0:
            cols = x_header_names[~categorical_values] if categorical_values else x_header_names
            for col in cols:
                df[col] = df[col].fillna(df.groupby(target)[col].transform('mean'))
        if categorical_values:
            df = pd.get_dummies(df, columns=categorical_values)

        reduc_func, norm_func = choose_procesing_methods(df.drop([target], 1))

        X = np.array(df.drop([target], 1))
        Y = np.array(df[target])

        X_scaled = scale_data(X, norm_func)
        reduc_X = reduc_data(x=X_scaled, y=Y, method=reduc_func)

        return (reduc_X, Y, x_header_names)
    else:
        if not data_file:
            return None
        else:
            if type(data_file) == dict:
                df = pd.DataFrame([data_file], columns=data_file.keys())
            else:
                df = pd.read_csv(data_file)
            df_init = pd.read_csv(initial_data)
        df_copy = copy.deepcopy(df)

        x_header_names = list(df_init.drop([target], 1))

        if df_init.isnull().sum().sum() != 0:
            cols = x_header_names[~categorical_values] if categorical_values else x_header_names
            for col in cols:
                df_init[col] = df_init[col].fillna(df_init.groupby(target)[col].transform('mean'))
        if categorical_values:
            df = pd.get_dummies(df, columns=categorical_values)
            df_init = pd.get_dummies(df_init, columns=categorical_values)
            df = pd.DataFrame(df, columns=df_init.columns.values[df_init.columns.values != target]).fillna(0)

        X_init = np.array(df_init.drop([target], 1))
        Y_init = np.array(df_init[target])
        X = np.array(df)

        reduc_func, norm_func = choose_procesing_methods(df_init.drop([target], 1))
        X_scaled = scale_data(X, norm_func, X_init)
        X_init_scaled = scale_data(X_init, norm_func)

        reduc_X = reduc_data(x=X_init_scaled, y=Y_init, method=reduc_func, x_to_predict=X_scaled)

        return (reduc_X, df_copy)
