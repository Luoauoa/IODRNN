import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from utils.data_utils import *
from statsmodels.tsa.vector_ar.var_model import VAR


input_time_step = 12
horizon = 1


def masked_smape(y_pred, y_true, null_value):
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    if np.isnan(null_value):
        mask = ~np.isnan(y_true)
    else:
        mask = (y_true != null_value)
    mask = mask.astype(np.float32)
    mask /= np.mean(mask)
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) * 0.5)
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return loss.mean()


def divergence(l1, l2):
    """
    :param l1: mat1 with shape (num_nodes, sequence_length)
    :param l2: mat2 with shape (num_nodes, sequence_length)
    :return:
    """
    divg = torch.sum(l1 * torch.log(l1 / (l2 + 1e-10) + 1e-10), dim=-1)
    return divg


def masked_diffmape(y_pred, y_true, type='div', epsilon=1e-10):
    # mask = (y_true == 0.)
    # y_true[mask] = epsilon
    # y_pred[mask] = epsilon
    # num_nodes, pre_len = y_pred.shape
    y_real = y_true[horizon:, :].T
    y_prev = y_true[:-horizon, :].T
    y_pred = y_pred[horizon:, :].T

    if type == 'div':
        t1 = torch.softmax(y_real, dim=-1)
        t2 = torch.softmax(y_prev, dim=-1)
        p = torch.softmax(y_pred, dim=-1)
        d1 = (divergence(p, t1) + divergence(t1, p))  # smaller means more similar
        d2 = (divergence(p, t2) + divergence(t2, p))
    else:
        d1 = None
        d2 = None
    weight = torch.mean(d1)
    diff = torch.exp(torch.mean((d1 - d2)))  # smaller is better
    loss = 2 * weight * diff / (weight + diff)
    return loss



def masked_mae(y_pred, y_true, null_value):
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    if np.isnan(null_value):
        mask = ~np.isnan(y_true)
    else:
        mask = (y_true != null_value)
    mask = mask.astype(np.float32)
    mask /= np.mean(mask)
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = np.abs(y_true - y_pred)
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return loss.mean()


def masked_mse(y_pred, y_true, null_value):
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    if np.isnan(null_value):
        mask = ~np.isnan(y_true)
    else:
        mask = (y_true != null_value)
    mask = mask.astype(np.float32)
    mask /= np.mean(mask)
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = np.power((y_true - y_pred), 2)
    loss = loss * mask
    loss = np.where(np.isnan(loss), np.zeros_like(loss), loss)
    return loss.mean()


def data_to_dataframe(data):
    """
    :param data: numpy with shape(feas, num_vers, num_timestep)
    :return: df: pandas.DataFrame, index: time, columns: sensor id, content: data.
    """
    n_f, n_v, n_t = data.shape
    index = data[0, 0, :].reshape(-1,)
    value = data[1, :, :].T
    columns = range(n_v)
    df = pd.DataFrame(data=value, index=index, columns=columns)

    return df


def var_predict(df, n_forwards=(1, 3), n_lags=12, test_ratio=0.2):
    """
    Multivariate time series forecasting using Vector Auto-Regressive Model.
    :param df: pandas.DataFrame, index: time, columns: sensor id, content: data.
    :param n_forwards: a tuple of horizons.
    :param n_lags: the order of the VAR model.
    :param test_ratio:
    :return: [list of prediction in different horizon], dt_test
    """
    n_sample, n_output = df.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    df_train, df_test = df.iloc[:n_train], df.iloc[n_train:]

    scaler = StandardScaler(mean=df_train.values.mean(), std=df_train.values.std())
    data = scaler.transform(df_train.values, category='data')
    var_model = VAR(data)
    var_result = var_model.fit(n_lags)
    max_n_forwards = np.max(n_forwards)
    # Do forecast
    result = np.zeros(shape=(len(n_forwards), n_test, n_output))
    start = n_train - n_lags - max_n_forwards + 1
    for input_ind in range(start, n_sample - n_lags):
        prediction = var_result.forecast(scaler.transform(df.values[input_ind: input_ind + n_lags], category='data'), max_n_forwards)
        for i, n_forward in enumerate(n_forwards):
            result_ind = input_ind - n_train + n_lags + n_forward - 1
            if 0 <= result_ind < n_test:
                result[i, result_ind, :] = prediction[n_forward - 1, :]

    df_predicts = []
    for i, n_forward in enumerate(n_forwards):
        df_predict = pd.DataFrame(scaler.inverse_transform(result[i]), index=df_test.index, columns=df_test.columns)
        df_predicts.append(df_predict)
    return df_predicts, df_test


if __name__ == '__main__':
    set_seed()
    X, _ = load_data()  # input data with shape(num_features, num_vertices, num_data)
    sprnn = np.load("sprnn_pre_1.npy", allow_pickle=True)
    #horizon = [3, 6, 12]
    # train_valid_test
    df = data_to_dataframe(X)
    df_pre, df_test = var_predict(df, n_forwards=(1, horizon), n_lags=12, test_ratio=0.2)
    predictions = torch.from_numpy(df_pre[1].values)
    #g_t = np.load("ground_true.npy", allow_pickle=True)
    #np.save("VAR_pre.npy", predictions)
    Y_test = torch.from_numpy(df_test.values)
    MAE = masked_mae(predictions, Y_test, 0.0)
    RMSE = np.sqrt(masked_mse(predictions, Y_test, 0.0))
    MAPE = masked_smape(predictions, Y_test, 0.0)
    Diff_MAPE = masked_diffmape(predictions, Y_test, type='div')
    print('MAE: {:.4}'.format(MAE))
    print('RMSE: {:.4}'.format(RMSE))
    print('MAPE: {:.4}'.format(MAPE))
    print('Diff_MAPE: {:.4}'.format(Diff_MAPE))
    plt.figure(figsize=(10, 6))
    plt.xlabel('Time')
    plt.ylabel('Traffic Flow')
    plt.plot(df_test.iloc[288 * 8:288 * 12, 1].values, label='Ground Truth')
    plt.plot(df_pre[0].iloc[288 * 8:288 * 12, 1].values, label='VAR')
    plt.plot(sprnn[288 * 8 - 12: 288 * 12 - 12], label='IODRNN')
    plt.legend(loc='upper left', ncol=3, fancybox=True)
    plt.grid()
    plt.show()

