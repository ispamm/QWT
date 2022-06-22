import munch
import torch

phm = False  # @param ["True", "False"] {type:"raw"}
qsn = True  # @param ["True", "False"] {type:"raw"}
real = False  # @param ["True", "False"] {type:"raw"}
soup = False  # @param ["True", "False"] {type:"raw"}
share_net_real = True  # @param ["True", "False"] {type:"raw"}
last_layer_gen_real = True  # @param ["True", "False"] {type:"raw"}
wavelet_disc_gen = (False, True) 
wavelet_type = "real" #@param ["real", "quat"]
spectral = True
experiment_name = "wavelet_gen_server"  # @param {type:"string"}
mode = "eval"  # @param ["train", "eval","sample"]
sepoch = 50  # @param {type:"integer"}
gpu_num = 0

args = munch.Munch({
    "mode": mode,
    "experiment_name": experiment_name,
    "datasets": "chaos",
    "dataset_path": "datasets/chaos2019/",
    "png_dataset_path": "datasets/chaos2019/png8020",
    "save_path": "pretrained_weights",
    "val_img_dir": "datasets/chaos2019/test",
    "eval_dir": "eval",
    "checkpoint_dir": "pretrained_weights/checkpoints/MICCAI2021/",
    "batch_size": 16,
    "eval_batch_size": 16,
    "gan_version": "Generator[2/3]+shapeunet+D",
    "image_size": 128,  # 256
    "epoch": 50,
    "sepoch": sepoch,
    "modals": ('t1', 't2', 'ct'),
    "lr": 1e-4,
    "loss_function": "wgan-gp+move+cycle+ugan+d+l2",
    "optimizer": "adam",
    "note": "affine:True;",
    "random_seed": 888,
    "log_every": 10,
    "print_every": 10,
    "save_every": 50,
    "c_dim": 3,
    "h_conv": 16,
    "G_conv": 64,
    "betas": (0.5, 0.9),
    "ttur": 3e-4,
    "w_d_false_c": 0.01,
    "w_d_false_t_c": 0.01,
    "w_g_c": 1.0,
    "w_g_t_c": 1.0,
    "w_g_cross": 50.0,
    "w_shape": 1,
    "w_cycle": 1,
    "phm": phm,
    "qsn": qsn,
    "real": real,
    "soup": soup,
    "last_layer_gen_real": last_layer_gen_real,
    "share_net_real": share_net_real,
    "wavelet_disc_gen": wavelet_disc_gen,
    "wavelet_type": wavelet_type,
    "spectral" : spectral,
    "seed": 888,
    "gpu_num": gpu_num
})
device = torch.device('cuda:' + str(args.gpu_num) if args.gpu_num > -1 else 'cpu')
