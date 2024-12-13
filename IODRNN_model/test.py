from engine import *
from utils.data_utils import set_seed
input_time_step = 12
output_time_step = 3


if __name__ == '__main__':
    torch.cuda.empty_cache()
    set_seed()
    engine_kwargs = {'batch_size': 64,
                     'epoch': 120,
                     'split_ratio': [0.7, 0.8],
                     'encode_layer_num': 2,
                     'decode_layer_num': 1,
                     'max_norm': 1.0,
                     'miles': [20, 30, 40, 50, 60],
                     'diffusion_step': 1,
                     'num_units': 64,
                     'lr_decay': True,
                     'decay_rate': 0.6,
                     'warmup_step': 20,
                     'init_lr': 2e-3,
                     'filter_type': 'Chebyshev',
                     'base_dir_log': 'saves/',
                     'log_name': 'processing.log'
                     }
    engine_box = Engine(**engine_kwargs)
    # testing
    model.load_state_dict(torch.load(os.path.join(path, 'mysprnn_{}'.format(epochs))))
    MAE, RMSE, sMAPE, diff = engine_box.test()
    print("Test over...")
    print("MAE: {:.4}".format(torch.mean(MAE)))
    print("RMSE: {:.4}".format(torch.mean(RMSE)))
    print("sMAPE: {:.4}".format(torch.mean(sMAPE)))
    print("diff: {:.4}".format(torch.mean(diff)))

    # visualization
    model_plot.load_state_dict(torch.load(os.path.join(path, 'mysprnn_{}'.format(epochs))))
    ground_true, predictions = predict_plot(X[:, :, split_valid:], input_time_step, output_time_step, model_plot)
    pre_path = os.path.join(path, 'sprnn_predictions_{}'.format(epochs))
    torch.save(predictions, pre_path)

    plt.figure(figsize=(10, 6))
    plt.plot(ground_true[288*1:288*5], label="Ground_Truth")
    plt.plot(predictions[288*1:288*5], label="Predict")
    plt.legend(loc='upper left', ncol=3, fancybox=True)
    plt.grid()
    plt.show()