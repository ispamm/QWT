import copy
import json
import os
import random

import munch

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
import torchvision.utils as vutils
# from importlib import reload
# import configs.config_tmp
# reload(configs.config_tmp)

# from configs.config_tmp import args, device
# print("utils module sees: ",args.seed)
# import model
# reload(model)
from config import args, device

from model import Discriminator, Generator, ShapeUNet

def loss_filter(mask, device="cuda"):
    lis = []
    for i, m in enumerate(mask):
        if torch.any(m == 1):
            lis.append(i)
    index = torch.tensor(lis, dtype=torch.long).to(device)
    return index


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


def denorm(x):
    res = (x + 1.) / 2.
    res.clamp_(0, 1)
    return res


def renorm(x):
    res = (x - 0.5) / 0.5
    res.clamp_(-1, 1)
    return res


def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    for i in range(batch_size):
        out[i, labels[i].long()] = 1
    return out


def getLabel(imgs, device, index, c_dim=2):
    syn_labels = torch.zeros((imgs.size(0), c_dim)).to(device)
    syn_labels[:, index] = 1.
    return syn_labels

 
def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]
    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
    return torch.mean((dydx_l2norm - 1) ** 2)


def save_state_net(net, parameters, index, optim=None, parents_root='checkpoints/MICCAI2021'):
    save_path = os.path.join(parameters.save_path, parents_root)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_file = os.path.join(save_path, parameters.net_name)
    torch.save(net.state_dict(), save_file + '_' + str(index) + args.experiment_name + '.pkl')
    if optim is not None:
        torch.save(optim.state_dict(), save_file + '_optim_' + str(index) + args.experiment_name + '.pkl')
    if not os.path.isfile(save_path + '/outputs.txt'):
        with open(save_path + '/outputs.txt', mode='w') as f:
            argsDict = parameters.__dict__;
            f.writelines(parameters.note + '\n')
            for i in argsDict.keys():
                f.writelines(str(i) + ' : ' + str(argsDict[i]) + '\n')


def load_state_net(net, net_name, index, optim=None, parents_root='checkpoints/MICCAI2021'):
    save_path = os.path.join(args.save_path, parents_root)
    if not os.path.isdir(save_path):
        raise Exception("wrong path")
    save_file = os.path.join(save_path, net_name)
    if net is not None:
        net.load_state_dict(torch.load(save_file + '_' + str(index) + args.experiment_name + '.pkl', map_location=device))
    if optim is not None:
        optim.load_state_dict(torch.load(save_file + '_optim_' + str(index) + args.experiment_name + '.pkl'))
    return net, optim


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def plot_images(netG_use, i, syneval_dataset, syneval_dataset2, syneval_dataset3):
    fig = plt.figure(dpi=120)
    with torch.no_grad():
        index = random.choice([0, 1, 2])
        if index == 0:
            img = syneval_dataset[13][0]
            mask = syneval_dataset[13][1].to(device)
        elif index == 1:
            img = syneval_dataset2[3][0]
            mask = syneval_dataset2[3][1].to(device)
        else:
            img = syneval_dataset3[43][0]
            mask = syneval_dataset3[43][1].to(device)
        img = img.unsqueeze(dim=0).to(device)
        # print(getLabel(img, device, 0, args.c_dim).shape,img.shape)
        pred_t1_img = netG_use(img, None, c=getLabel(img, device, 0, args.c_dim), mode='test')
        pred_t2_img = netG_use(img, None, c=getLabel(img, device, 1, args.c_dim), mode='test')
        pred_t3_img = netG_use(img, None, c=getLabel(img, device, 2, args.c_dim), mode='test')
        plt.subplot(241)
        plt.imshow(denorm(img).squeeze().cpu().numpy(), cmap='gray')
        plt.title(str(i + 1) + '_source')
        plt.subplot(242)
        plt.imshow(denorm(pred_t1_img).squeeze().cpu().numpy(), cmap='gray')
        plt.title('pred_x1')
        plt.subplot(243)
        plt.imshow(denorm(pred_t2_img).squeeze().cpu().numpy(), cmap='gray')
        plt.title('pred_x2')
        plt.subplot(244)
        plt.imshow(denorm(pred_t3_img).squeeze().cpu().numpy(), cmap='gray')
        plt.title('pred_x3')
        plt.show()
        x_concat = [denorm(img).squeeze().cpu(),
                    denorm(pred_t1_img).squeeze().cpu(),
                    denorm(pred_t2_img).squeeze().cpu(),
                    denorm(pred_t3_img).squeeze().cpu()]
        x_concat = torch.cat(x_concat, dim=0)
        plt.close(fig)
    return x_concat


def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def save_image(x, ncol, filename):
    x = denorm(x)
    # IF BATCH
    # if args.mode=="train":
    #     iters = str(int(filename.replace(args.experiment_name,"").split("/")[8].split("_")[0]))
    # # IS ONLY ONE PHOTO
    # iters = str(int(filename.split("/")[9].split("_")[0]))
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)
    if len(x.shape) == 4 and args.mode == "train":
        wandb.log({'sample_dir': wandb.Image(filename, caption=iters)}, commit=False)

def load_nets(nets):
    for net in nets.keys():
        print("loading", net)
        net_check = net if "use" in net else net.replace("_", "")
        print(args.sepoch)
        load_state_net(nets[net], net_check, args.sepoch)


def build_model():
    if not args.real and args.soup:
        disc_c_dim = 8
    elif args.real:
        disc_c_dim = args.c_dim * 2
    else:
        #wavelets
        disc_c_dim = 6
    channels = 5 if (
        (
            not args.real and 
            not args.soup and 
            args.wavelet_disc_gen[1] and
            not args.wavelet_net 
        ) 
        or args.wavelet_with_real_net or args.wavelet_disc_gen[1]) else 1
    if args.is_best_4 and args.wavelet_type=='quat' and args.wavelet_disc_gen[1]:
        channels = 1+len(args.best_4)
    netG = Generator(in_c=channels + args.c_dim, mid_c=args.G_conv, layers=2, s_layers=3, affine=True, last_ac=True).to(
        device)



    netD_i = Discriminator(c_dim=disc_c_dim, image_size=args.image_size).to(device)
    if args.target_real:
        args.qsn= False
        args.real = True
        shape_net_channels = 4 if not args.real and args.wavelet_disc_gen[2] else 1
        netH = ShapeUNet(img_ch=shape_net_channels, mid=args.h_conv, output_ch=shape_net_channels).to(device)
        netD_t = Discriminator(c_dim=disc_c_dim, image_size=args.image_size, target = True).to(device)
        args.real = False
        args.qsn = True
    else:
        #TODO
        shape_net_channels = 1 if args.last_layer_gen_real else 4
        netH = ShapeUNet(img_ch=shape_net_channels, mid=args.h_conv, output_ch=shape_net_channels).to(device)
        netD_t = Discriminator(c_dim=disc_c_dim, image_size=args.image_size).to(device)

    if args.shape_network_sep_target:
        if args.target_real:
            args.qsn= False
            args.real = True
        netH_t = ShapeUNet(img_ch=shape_net_channels, mid=args.h_conv, output_ch=shape_net_channels).to(device)
        if args.target_real:
            args.real = False
            args.qsn = True
    else:
        netH_t = None

    netG_use = copy.deepcopy(netG)
    netG.to(device)
    netD_i.to(device)
    netD_t.to(device)
    netH.to(device)
    netG_use.to(device)
    nets = munch.Munch({"netG": netG,
                        "netD_i": netD_i,
                        "netD_t": netD_t,
                        "netH": netH,
                        "netH_t": netH_t,
                        "netG_use": netG_use})
    return nets, disc_c_dim


def build_optims(nets, glr, dlr):
    g_optimizier = torch.optim.Adam(nets.netG.parameters(), lr=glr, betas=(args.betas[0], args.betas[1]))
    di_optimizier = torch.optim.Adam(nets.netD_i.parameters(), lr=dlr, betas=(args.betas[0], args.betas[1]))
    dt_optimizier = torch.optim.Adam(nets.netD_t.parameters(), lr=dlr, betas=(args.betas[0], args.betas[1]))
    h_optimizier = torch.optim.Adam(nets.netH.parameters(), lr=glr, betas=(args.betas[0], args.betas[1]))
    if args.shape_network_sep_target:
        h_t_optimizier = torch.optim.Adam(nets.netH_t.parameters(), lr=glr, betas=(args.betas[0], args.betas[1]))
    else:
        h_t_optimizier = None
    return munch.Munch({"g_optimizier": g_optimizier, "di_optimizier": di_optimizier, "dt_optimizier": dt_optimizier,
                        "h_optimizier": h_optimizier, "h_t_optimizier": h_t_optimizier})


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))
    return num_params
