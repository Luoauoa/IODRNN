from engine import *
from utils.data_utils import set_seed
device = None
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

if __name__ == '__main__':
    # 清除cuda显存
    torch.cuda.empty_cache()
    set_seed()
    save_losses = False
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
                     'log_name': 'processing.log',
                     'add': False
                     }
    engine_box = Engine(**engine_kwargs)
    # train & valid
    train_losses, valid_losses = engine_box.train()
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='training_loss')
    plt.plot(valid_losses, label='validation_loss')
    plt.legend()
    plt.title('Loss Variation')
    plt.show()
    # save validation losses
    if save_losses:
        loss_path = os.path.join(engine_kwargs['base_dir_log'], 'val_loss{}'.format(engine_kwargs['epoch']))
        torch.save(valid_losses, loss_path)

