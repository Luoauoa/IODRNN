from sklearn.svm import LinearSVR
from utils.data_utils import *
import matplotlib.pyplot as plt
input_time_step = 12
output_time_step = 3
set_seed()


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
    # y_pred = y_pred.permute(2, 0, 1)  # (batch_size, pre_len, num_nodes) -> (num_nodes, batch_size, pre_len)
    # y_true = y_true.permute(2, 0, 1)
    # mask = (y_true == 0.)
    # y_true[mask] = epsilon
    # y_pred[mask] = epsilon
    num_nodes, sample, pre_len = y_pred.shape
    y_real = y_true[:, 1:, :].reshape(num_nodes, -1)
    y_prev = y_true[:, :-1, :].reshape(num_nodes, -1)
    y_pred = y_pred[:, 1:, :].reshape(num_nodes, -1)
    assert y_real.shape == (num_nodes, (sample - 1) * pre_len)

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
    diff = torch.exp(torch.mean(d1 - d2))  # smaller is better
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


if __name__ == '__main__':
    set_seed()
    X, _ = load_data()  # input data with shape(num_features, num_vertices, num_data)

    # train_valid_test——
    split_train = int(X.shape[2] * 0.7)
    split_valid = int(X.shape[2] * 0.8)
    data_dict, target_dict, scaler = generate_dataset(X, num_timestep_input=input_time_step,
                                                      num_timestep_output=output_time_step,
                                                      split=[split_train, split_valid])
    num_nodes = 100

    svr = LinearSVR(verbose=True, C=1.0)
    predictions = []
    for n in range(num_nodes):
        X_train = data_dict['train'][:, :, n, 1].numpy()  # (num_data, input_len)
        Y_train = target_dict['train'][:, :, n].numpy()  # (num_data, out_len)
        Y_train = np.mean(Y_train, axis=-1)
        X_test = data_dict['test'][:, :, n, 1].numpy()

        model = svr.fit(X_train, Y_train)
        pre = np.expand_dims(svr.predict(X_test), axis=-1)  # (num_data(test), 1)
        assert pre.shape == (X_test.shape[0], 1)
        pre = pre.repeat(output_time_step, axis=-1)
        predictions.append(pre)
    Y_test = target_dict['test'].permute((2, 0, 1))  # (num_nodes, num_data(test), out_len)
    predictions = torch.from_numpy(np.array(predictions))  # (num_nodes, num_data(test), out_len)
    # Y_test = scaler.inverse_transform(Y_test)
    # predictions = scaler.inverse_transform(predictions)

    MAE = masked_mae(predictions, Y_test, 0.0)
    RMSE = np.sqrt(masked_mse(predictions, Y_test, 0.0))
    MAPE = masked_smape(predictions, Y_test, 0.0)
    Diff_MAPE = masked_diffmape(predictions, Y_test, type='div')
    print('MAE: {:.4}'.format(MAE))
    print('RMSE: {:.4}'.format(RMSE))
    print('MAPE: {:.4}'.format(MAPE))
    print('Diff_MAPE: {:.4}'.format(Diff_MAPE))
    plt.plot(predictions[0].flatten())
    # plt.legend()
    plt.show()
