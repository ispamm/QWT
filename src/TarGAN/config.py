import munch
import torch
from torchvision import transforms
phm = False  # @param ["True", "False"] {type:"raw"}
qsn = False  # @param ["True", "False"] {type:"raw"}
real = True  # @param ["True", "False"] {type:"raw"}
soup = False  # @param ["True", "False"] {type:"raw"}
share_net_real = True  # @param ["True", "False"] {type:"raw"}
last_layer_gen_real = True  # @param ["True", "False"] {type:"raw"}

#wavelets config
wavelet_with_real_net = False
wavelet_disc_gen = (False, True, False) #disc gen shape_controller
wavelet_target = False #wavelet also on segmented image
wavelet_type = "q" #@param ["real", "quat", 'fusion']
wavelet_quat_type = "low" #low or whatever (ele method)
wavelet_net = False
wavelet_net_real = False
wavelet_net_target = False
wavelet_net_target_real = False

spectral = True
target_real = False
shape_network_sep_target = False
is_best_4 = False
best_4 = [0,1]

t1_best_4 = [0, 4, 10, 11]
t2_best_4 = [0, 4, 5, 10]
ct_best_4 = [0, 2, 3, 4]
seed = 1761017
experiment_name = "qwtargan_inverted_novel"+str(seed) #"qwtargan_novel_best_"+str(len(best_4))+"_moe"+str(seed)  # @param {type:"string"}
mode = "train"  # @param ["train", "eval","sample"]
sepoch = 0  # @param {type:"integer"}
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
    "image_size": 256,  # 256
    "epoch": 50,
    "sepoch": sepoch,
    "modals": ('t1', 't2', 'ct'),
    "lr": 1e-4,
    "loss_function": "wgan-gp+move+cycle+ugan+d+l2",
    "optimizer": "adam",
    "note": "affine:True;",
    "log_every": 10,
    "print_every": 10,
    "save_every": 50,
    "eval_every": 50,
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
    "wavelet_with_real_net": wavelet_with_real_net,
    "wavelet_disc_gen": wavelet_disc_gen,
    "wavelet_type": wavelet_type,
    "wavelet_quat_type": wavelet_quat_type,
    'wavelet_target':wavelet_target,
    "wavelet_net": wavelet_net,
    "wavelet_net_real": wavelet_net_real,
    "wavelet_net_target": wavelet_net_target,
    "wavelet_net_target_real": wavelet_net_target_real,
    "spectral" : spectral,
    "target_real": target_real,
    "shape_network_sep_target": shape_network_sep_target,
    "best_4": best_4,
    "t1_best_4": t1_best_4,
    "t2_best_4": t2_best_4,
    "ct_best_4": ct_best_4,
    "is_best_4": is_best_4,
    "seed": seed,
    "gpu_num": gpu_num
})

grayscale = transforms.Grayscale(num_output_channels=1)

device = torch.device('cuda:' + str(args.gpu_num) if torch.cuda.is_available() else 'cpu')
