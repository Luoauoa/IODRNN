import os
import numpy as np
import matplotlib.pyplot as plt
import torch

device = ("cuda:0" if torch.cuda.is_available() else 'cpu')


class Dataloader:

    def __init__(self, data_dict, target_dict, batch_size, pad_with_last, shuffle, mode):
        self.data_dict = data_dict
        self.target_dict = target_dict
        self.batch_size = batch_size
        self.pad_with_last = pad_with_last
        self.shuffle = shuffle
        self.mode = mode
        self.X = self.data_dict[self.mode]   # tensor
        self.Y = self.target_dict[self.mode]
        self.total_batch = 0
        # pad or not
        X = self.X
        Y = self.Y
        if self.pad_with_last:
            pad_length = (self.batch_size - (len(X) % self.batch_size)) % self.batch_size
            x_padding = np.repeat(X[-1:], pad_length, axis=0)
            y_padding = np.repeat(Y[-1:], pad_length, axis=0)
            X = np.concatenate([X, x_padding], axis=0)
            Y = np.concatenate([Y, y_padding], axis=0)
            self.total_batch = int(X.shape[0] // self.batch_size)

        # shuffle or not
        if self.shuffle:
            indices = torch.randperm(X.shape[0])
            X = X[indices]
            Y = Y[indices]
        self.X = X
        self.Y = Y

    def _wrapper(self):
        x_batch = []
        y_batch = []
        for i in range(0, self.X.shape[0], self.batch_size):
            x_batch.append(self.X[i:i+self.batch_size])
            y_batch.append(self.Y[i:i+self.batch_size])

        x_batch = torch.from_numpy(np.array(x_batch))
        assert x_batch.shape[1] == self.batch_size
        y_batch = torch.from_numpy(np.array(y_batch))
        assert y_batch.shape[1] == self.batch_size
        return self.total_batch, x_batch, y_batch


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_smape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds-labels))/((torch.abs(labels) + torch.abs(preds)) * 0.5)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def divergence(l1, l2):
    """
    :param l1: mat1 with shape (num_nodes, sequence_length)
    :param l2: mat2 with shape (num_nodes, sequence_length)
    :return:
    """
    divg = torch.sum(l1 * torch.log(l1 / (l2 + 1e-12) + 1e-12), dim=-1)
    return divg


def cosine_similarity(m1, m2):
    """

    :param m1: mat1 with shape (num_nodes, sequence_length)
    :param m2: mat2 with shape (num_nodes, sequence_length)
    :return:
    """
    dot = torch.sum(m1 * m2, dim=-1)  # (num_nodes, 1)
    norm = torch.sqrt(torch.sum(m1 ** 2, dim=-1) * torch.sum(m2 ** 2, dim=-1))
    cos_dis = dot / norm
    return cos_dis


def diff_similarity(y_pred, y_true, type='cos'):
    # implementation of SDD - shift difference distance
    y_pred = y_pred.permute(2, 0, 1)  # (batch_size, pre_len, num_nodes) -> (num_nodes, batch_size, pre_len)
    y_true = y_true.permute(2, 0, 1)
    num_nodes, sample, pre_len = y_pred.shape
    y_real = y_true[:, 1:, :].reshape(num_nodes, -1)
    y_prev = y_true[:, :-1, :].reshape(num_nodes, -1)
    y_pred = y_pred[:, 1:, :].reshape(num_nodes, -1)
    assert y_real.shape == (num_nodes, (sample - 1) * pre_len)
    weight = 0
    loss = None

    if type == 'cos':
        real_sim = cosine_similarity(y_pred, y_real)
        prev_sim = cosine_similarity(y_pred, y_prev)  # higher means more similar
        weight = real_sim + prev_sim
        loss = (prev_sim - real_sim)  # similarity, smaller is better

    if type == 'div':
        t1 = torch.softmax(y_real, dim=-1)
        t2 = torch.softmax(y_prev, dim=-1)
        p = torch.softmax(y_pred, dim=-1)
        d1 = divergence(p, t1) + divergence(t1, p)  # smaller means more similar
        d2 = divergence(p, t2) + divergence(t2, p)
        weight = (d1 + d2) * 0.5
        loss = torch.mean((d1 - d2) / d2)  # smaller is better
        loss = torch.exp(loss) * weight
    return weight, loss


class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=device))
            torch.nn.init.xavier_normal_(nn_param)
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
            self._params_dict[shape] = nn_param

        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]

