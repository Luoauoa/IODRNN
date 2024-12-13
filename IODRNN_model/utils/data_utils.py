import numpy as np
import random
import os
import torch
import scipy.sparse as sp
import zipfile

data_path = '../data_new/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=99):
    """
    Set seed in case of variant results while doing experiments.
    :param seed:
    :return:
    """
    # os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


class StandardScaler:
    """
    Standard the inputs using Z-score method.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data, category):
        if category == 'data':
            return (data - self.mean) / self.std
        else:
            return (data - self.mean[..., 1]) / self.std[..., 1]

    def inverse_transform(self, data):
        # if data.device == device:
        #     data = data.cpu()
        data_gpu = (data * self.std + self.mean)
        return data_gpu


def get_normalized_adj(A):
    """
    return normalized adjacent matrix
    :param A: adj matrix A with shape
    :return: normalized adj matrix A_wave
    """
    # A = sp.coo_matrix(A)
    A = np.eye(A.shape[0], dtype=np.float32) + A
    D = np.sum(A, axis=1).reshape((-1,))
    D[D <= 1e-5] = 1e-5
    d_norm = np.reciprocal(np.sqrt(D))  # D ^ -1/2, (1-D array)
    return d_norm * A * d_norm


def get_asy_adj(adj):
    """
    return an asymmetric matrix
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(row_sum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def load_data():
    if(not os.path.isfile(os.path.join(data_path, "adj_mat.npy"))
            or not os.path.isfile(os.path.join(data_path, "nodes_ori.npy"))):
        exit(0)

    A = np.load(os.path.join(data_path, "adj_mat.npy"), allow_pickle=True)
    A = A.astype(np.float32)
    X = np.load(os.path.join(data_path, "nodes_ori.npy"), allow_pickle=True)  # (num_vertices, num_data, num_features)
    X = X.transpose((2, 0, 1))  # (num_features, num_vertices, num_data)
    X = X.astype(np.float32)

    return X, A


def generate_dataset(X, num_timestep_input, num_timestep_output, split):
    """
    generates datasets could be feeded into model.
    divides it into multiple samples along the time_axis by sliding a window
    of size (num_timestep_input+num_timestep_output) across it in step of 1.

    :param num_timestep_output:
    :param num_timestep_input:
    :param X:
            original data with shape(num_features, num_vertices, num_data(time_step))
    :param split:
            train, valid splitting ratio.
    :return:
            data: train data X of size (n_samples, time_step_input, num_vertices, num_features)
            target: target y of size (n_samples, time_step_output, num_vertices)
    """
    # generate begin and end indices of samples each of which contains
    # (num_train + num_target) points
    data_dict = {}
    target_dict = {}
    indices = [(i, i + num_timestep_input + num_timestep_output)
               for i in range(X.shape[2] - (num_timestep_input + num_timestep_output) + 1)]

    # Get and save samples
    data = []
    target = []

    # feature(time, flow, num_gps, zero_speed, period)
    for i, j in indices:
        data.append(X[:, :, i: i + num_timestep_input].transpose((2, 1, 0)))  # (num_data, num_nodes, num_feature)
        target.append(X[1, :, i + num_timestep_input: j].transpose((1, 0)))
    data = np.array(data)
    target = np.array(target)
    data_dict['train'], target_dict['train'] = torch.from_numpy(data[:split[0]]), torch.from_numpy(target[:split[0]])
    data_dict['valid'], target_dict['valid'] = torch.from_numpy(data[split[0]:split[1]]), torch.from_numpy(target[split[0]:split[1]])
    data_dict['test'], target_dict['test'] = torch.from_numpy(data[split[1]:]), torch.from_numpy(target[split[1]:])

    scaler = StandardScaler(torch.nanmean(data_dict['train'], dim=(0, 1, 2)).reshape(1, 1, 1, -1),
                            torch.std(data_dict['train'], dim=(0, 1, 2)).reshape(1, 1, 1, -1))
    for category in ['train', 'valid', 'test']:
        data_dict[category] = scaler.transform(data_dict[category], 'data')
    return data_dict, target_dict, scaler


if __name__ == '__main__':
    # statistics
    X, A = load_data()
    # print(X.shape)
    import matplotlib.pyplot as plt
    flow = X[1, :, :].reshape(-1,)
    speed = np.array([22.68631752, 22.85292197, 22.27754851, 22.45408637, 23.05128381,
       22.88104657, 22.66733289, 23.44967119, 23.56200137, 17.21445439,
       22.31619492, 22.62849635, 21.83099187, 21.02596346, 21.87556476,
       22.4316118 , 22.32154875, 25.67358815, 22.46600716, 21.67758459,
       21.14898115, 22.20932025, 22.37592153, 21.59726507, 21.82612083,
       22.26364216, 21.91350256, 22.40176768, 22.43747257, 21.3836073 ,
       21.17738574, 22.03547445, 22.15660784, 21.78777688, 21.80516931,
       21.25664048, 21.65149003, 21.70180858, 22.27300199, 21.75082711,
       22.47112127, 21.3290068, 21.3568454 , 21.11439269, 20.2198194 ,
       20.81077311, 21.03919214, 21.84336411, 21.14559476, 21.69109686,
       22.8980111 , 22.45657719, 21.82962971, 22.13965223, 21.1592355 ,
       20.55825661, 20.23173965, 21.06071401, 21.61697467, 22.73070529,
       22.86339785, 22.75280602, 22.76525173, 23.39105375, 22.08671189,
       20.99241544, 22.79626373, 22.65024186, 21.93981963, 21.77442805,
       23.32487843, 23.36824524, 21.99016202, 23.09374441, 23.22218186,
       22.33807602, 21.85418329, 22.26883142, 21.78225795, 22.52669331,
       25.61348185, 24.14283505, 26.03889239, 23.17888114, 23.32026987,
       22.20293826, 21.30733483, 22.21663192, 22.15559889, 23.26656342,
       22.70841892, 27.7589433 , 32.0248469 , 24.55212541, 23.64546656,
       22.39069951, 22.38194486, 22.19971093, 22.7363918, 22.4843024]).reshape(-1,)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    bplot_flow = ax[0, 0].boxplot(flow, vert=True, patch_artist=True, whiskerprops={'linestyle':'--'}, showmeans=True)
    violin_flow = ax[0, 1].violinplot(flow, showmeans=True, showmedians=True, showextrema=True)
    bplot_speed = ax[1, 0].boxplot(speed, vert=True, patch_artist=True, whiskerprops={'linestyle':'--'}, showmeans=True)
    violin_speed = ax[1, 1].violinplot(speed, showmeans=True, showmedians=True, showextrema=True)
    # ax1.set_xlabel("Traffic Flow", fontsize=12)
    plt.grid()
    ax[0, 0].set_xticks([])
    ax[1, 0].set_xticks([])
    ax[1, 1].set_xticks([])
    ax[0, 1].set_xticks([])
    fig.suptitle("Traffic Flow per minute\n mean={0:.2f}    std={1:.2f}".format(np.mean(flow), np.std(flow)), fontsize=12)
    print("Traffic Flow per minute\n mean={0:.2f}    std={1:.2f}".format(np.mean(speed), np.std(speed)))
    plt.tight_layout()
    plt.show()
