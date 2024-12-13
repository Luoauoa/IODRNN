import logging

import torch.optim as optim
import sys
import os
import time
from tqdm import tqdm

import utils.logger_utils
from utils.data_utils import *
from utils.train_utils import *
from SpeculateModel import SpeculateGCRNN

input_time_step = 12
output_time_step = 3

device = None
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print('use device: %s' % device)


class Engine:
    def __init__(self, **kwargs):
        """"""
        self.kwargs = kwargs
        X, adj_mx = load_data()
        self.X = X
        self.adj_mx = adj_mx
        split_ratio = self.kwargs['split_ratio']
        split = [round(split_ratio[0] * X.shape[-1]), round(split_ratio[1] * X.shape[-1])]
        self.data_dict, self.target_dict, self.scaler = generate_dataset(self.X, num_timestep_input=input_time_step,
                                                                         num_timestep_output=output_time_step,
                                                                         split=split)

        # train & valid setting
        self.batch_size = self.kwargs.get('batch_size')
        encoder_shape_train = (self.batch_size, self.X.shape[1], self.X.shape[0])
        decoder_shape_train = (self.batch_size, self.X.shape[1], 1)
        encoder_shape_test = (1, self.X.shape[1], self.X.shape[0])  # (batch_size, num_nodes, num_features)
        decoder_shape_test = (1, self.X.shape[1], 1)
        self.log_name = self.kwargs.get('log_name')
        self.log_dir = self._get_log_dir(kwargs)
        self._logger = utils.logger_utils.get_logger(log_dir=self.log_dir, name=__name__,
                                                     level=logging.INFO, filename=self.log_name)
        self.epoch = self.kwargs['epoch']
        self.max_norm = self.kwargs['max_norm']
        self.miles = self.kwargs['miles']
        self.init_lr = self.kwargs['init_lr']
        self.decay_rate = self.kwargs['decay_rate']
        self.lr_decay = self.kwargs['lr_decay']
        self.warmup_step = self.kwargs['warmup_step']
        self.add = self.kwargs['add']
        # initial models
        self.train_model = SpeculateGCRNN(
            encoder_shape=encoder_shape_train,
            decoder_shape=decoder_shape_train,
            encode_layer=self.kwargs['encode_layer_num'],
            decode_layer=self.kwargs['decode_layer_num'],
            input_time_step=input_time_step,
            output_time_step=output_time_step,
            num_units=self.kwargs['num_units'],
            adj_mx=self.adj_mx,
            diffusion_step=self.kwargs['diffusion_step'],
            add=self.add
        ).to(device)

        # model for visualizing
        self.test_model = SpeculateGCRNN(
            encoder_shape=encoder_shape_test,
            decoder_shape=decoder_shape_test,
            encode_layer=self.kwargs['encode_layer_num'],
            decode_layer=self.kwargs['decode_layer_num'],
            input_time_step=input_time_step,
            output_time_step=output_time_step,
            num_units=self.kwargs['num_units'],
            adj_mx=self.adj_mx,
            diffusion_step=self.kwargs['diffusion_step'],
            add=self.add
        ).to(device)

    @staticmethod
    def _get_log_dir(kwargs):
        batch_size = kwargs.get('batch_size')
        epoch_num = kwargs.get('epoch')
        init_lr = kwargs.get('init_lr')
        diffusion_step = kwargs.get('diffusion_step')
        units = kwargs.get('num_units')
        filter_type = ('C' if kwargs.get('filter_type') == 'Chebyshev' else 'D')
        horizon = output_time_step
        model_id = 'sprnn_%s%d_h_%d_lr_%g_bs_%d_u_%d_epo_%d_%s/' % (
            filter_type, diffusion_step, horizon,
            init_lr, batch_size, units, epoch_num,
            time.strftime('%m%d-%H%M'))
        base_dir_log = kwargs.get('base_dir_log')
        log_dir = os.path.join(base_dir_log, model_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def train(self, **kwargs):
        kwargs.update(self.kwargs)
        return self._train(**kwargs)

    def test(self, **kwargs):
        kwargs.update(self.kwargs)
        return self._test(**kwargs)

    def _train_epoch(self, model, optimizer, epoch_num, mode):
        """
        train dcrnn model
        :param mode: which mode: train or val
        :param model: stgcn model
        :param optimizer: optimizer(Adam)
        :param epoch_num: which epoch am I now

        :return: average loss of this epoch
        """
        batch_size = self.batch_size
        # select train or valid data
        data_loader = Dataloader(data_dict=self.data_dict, target_dict=self.target_dict,
                                 batch_size=batch_size, pad_with_last=True,
                                 shuffle=False, mode=mode)
        total, X, Y = data_loader._wrapper()
        epoch_losses = []
        with tqdm(total=total, desc=f'Epoch {epoch_num + 1}/{self.epoch}', file=sys.stdout) as pbar:
            for x_batch, y_batch in zip(X, Y):
                batches_seen = batch_size * epoch_num
                # train or valid
                is_train = (True if mode == 'train' else False)
                model.train(is_train)
                x_batch = x_batch.to(device)  # (batch_size, input_step, num_nodes, num_features)
                y_batch = y_batch.to(device)  # (batch_size, output_step, num_nodes)
                labels = y_batch.permute(1, 0, 2)  # curriculum_learning labels
                with torch.set_grad_enabled(is_train):
                    if is_train:
                        out = model(x_batch, batches_seen, labels, is_train=True)
                    else:
                        out = model(x_batch, batches_seen, is_train=is_train)
                    loss = masked_mae(self.scaler.inverse_transform(out), y_batch, 0.0)
                # if train, backprop and update params
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_norm)
                    optimizer.step()
                    batches_seen += 1
                epoch_losses.append(loss.detach().cpu().numpy())
                pbar.update()
            pbar.close()
        return np.mean(epoch_losses)

    def _train(self, **kwargs):
        model = self.train_model
        init_lr = self.init_lr
        optimizer = optim.Adam(model.parameters(), lr=init_lr)
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                           milestones=self.miles, gamma=self.decay_rate)
        train_losses = []
        valid_losses = []
        self._logger.info("Start Training......")
        for i in range(self.epoch):
            self._logger.info("Epoch {0}/{1}:".format(i, self.epoch))
            if i <= self.warmup_step:
                warmup_percent_done = i / self.warmup_step
                warmup_learning_rate = init_lr * warmup_percent_done ** 5  # gradual warmup_lr
                lr = warmup_learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            train_loss = self._train_epoch(model, optimizer, i, mode='train')
            valid_loss = self._train_epoch(model, optimizer, i, mode='valid')
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            # decay the learning rate
            if self.lr_decay:
                lr_schedule.step()
            # if i % 5 == 0:
            # print('Training loss {0}, Validation loss {1}\n'.format(train_loss, valid_loss), flush=True)
            self._logger.info('epoch done......')
            self._logger.info('Training loss {0}, Validation loss {1}\n'.format(train_loss, valid_loss))
        self.save_model(self.epoch)
        return train_losses, valid_losses

    def _test(self, **kwargs):
        """
        Testing and returning metrics
        :return: tensors: MAE, RMSE, sMAPE， Diff
        """
        # load model
        model = self.train_model
        self.load_model(model)
        model.eval()
        loader = Dataloader(self.data_dict, self.target_dict, self.batch_size,
                            pad_with_last=True, shuffle=False, mode='test')
        total, X, Y = loader._wrapper()
        MAE = []
        RMSE = []
        sMAPE = []
        diff = []
        self._logger.info('Start evaluating......')
        with torch.no_grad():
            for x_batch, y_batch in zip(X, Y):
                x_batch = x_batch.to(device=device)
                y_hat = self.scaler.inverse_transform(model(x_batch, is_train=False))
                MAE.append(masked_mae(y_hat, y_batch, 0.0))
                RMSE.append(torch.sqrt(masked_mse(y_hat, y_batch, 0.0)))
                sMAPE.append(masked_smape(y_hat, y_batch, 0.0))
                weight, diff_batch = diff_similarity(y_hat, y_batch, type='div')   # SDD
                diff.append(diff_batch)
        MAE = torch.mean(torch.stack(MAE))
        RMSE = torch.mean(torch.stack(RMSE))
        sMAPE = torch.mean(torch.stack(sMAPE))
        diff = torch.mean(torch.stack(diff))
        self._logger.info('evaluating over and metrics are follows:\n MAE: {:.4f}'
                          '\nRMSE: {:.4f}\nsMAPE: {:.4f}\nSDD: {:.4f}'.format(MAE, RMSE, sMAPE, diff))
        return MAE, RMSE, sMAPE, diff

    def predict_plot(self):
        """
        predicting and returning (ground_truth, predictions)
        :param model:
        :return: (ground_truth, predictions)
        """
        # prepare plot data
        ground_truth = self.X[0, :, input_time_step:][0].reshape(-1, ).detach().cpu()  # single nodes
        # ground_truth = np.mean(X[1, :, num_timestep_input:], axis=0).reshape(-1,)  # average speed of all nodes
        predictions = []
        data = self.get_plot_data()
        # load model
        model = self.test_model
        self.load_model(model)
        model.eval()
        with tqdm(total=data.shape[0], desc=f"Epoch{1}/{1}", file=sys.stdout) as pbar:
            with torch.no_grad():
                for x in data:
                    x = x.unsqueeze(dim=0)  # (batch_size=1, num_timestep_input, num_nodes, num_features)
                    x = x.to(device)
                    prediction = model(x, is_train=False)  # (batch_size=1, out_time_step, num_nodes * out_dim)
                    predictions.append(prediction.detach().cpu())  # (n_samples, 1, out_time_step, num_nodes)
                    pbar.update()
                pbar.close()
        predictions = torch.stack(predictions).to(device)
        predictions = self.scaler.inverse_transform(predictions)[:, :, :, 0]
        # predictions = torch.mean(scaler.inverse_transform(predictions), dim=-1)  # mean of all nodes
        predictions = predictions.reshape(-1,).detach().cpu()

        # visualizing
        plt.figure(figsize=(10, 6))
        plt.plot(ground_truth[288 * 1:288 * 5], label="Ground_Truth")
        plt.plot(predictions[288 * 1:288 * 5], label="Predict")
        plt.legend(loc='upper left', ncol=3, fancybox=True)
        plt.grid()
        plt.show()

    def save_model(self, epoch):
        if not os.path.exists('saves/'):
            os.makedirs('saves/')
        model_state = self.train_model.state_dict()
        torch.save(model_state, 'saves/sprnn/epo%d.tar' % epoch)
        self._logger.info('Saved model at {}'.format(epoch))
        return 'saves/sprnn/epo%d.tar' % epoch

    def load_model(self, model):
        """
        :param model:self.train_model for testing, self.test_model for plotting
        :return:
        """
        assert os.path.exists('saves/sprnn/epo%d.tar' % self.epoch), 'Weights at epoch %d not found' % self.epoch
        checkpoint = torch.load('saves/sprnn/epo%d.tar' % self.epoch, map_location='cpu')
        model.load_state_dict(checkpoint)
        self._logger.info("Loaded model at {}".format(self.epoch))

    def get_plot_data(self):
        X = self.X
        data = []
        indices = [(i, i + input_time_step) for i in range(0, X.shape[-1] - input_time_step, output_time_step)]
        for i, j in indices:
            data.append(X[:, :, i:j].permute((2, 1, 0)))  # list of ndarray(timestep_input, num_nodes, num_features)
        data = torch.stack(data)  # tensor(n_samples, timestep_input, num_nodes, num_features)
        data = self.scaler.transform(data, 'data')
        return data

