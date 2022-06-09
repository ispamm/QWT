import time
import datetime

import munch
import torch
import torchvision
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import args, device
from dataset import ChaosDataset_Syn_new, ChaosDataset_Syn_Test
from metrics import calculate_all_metrics
from model import Generator, ShapeUNet, Discriminator
from utils import set_seed, load_state_net, loss_filter, label2onehot, gradient_penalty, denorm, moving_average, \
    plot_images, save_state_net
import copy
import torch.nn.functional as F


def convert_data_for_quaternion_tarGAN(batch):
    """
    converts batches of black and white images in 4 channels for QNNs
    """
    grayscale = torchvision.transforms.Grayscale(num_output_channels=1)
    x_real, t_img, shape_mask, mask, label_org = [], [], [], [], []
    # (x_real, t_img, shape_mask, mask, label_org)
    for i in range(len(batch)):
        _x_real = batch[i][0].repeat(3, 1, 1)
        _t_img = batch[i][1].repeat(3, 1, 1)
        # assert all(batch[i][0].size(0) == 3 for i in range(len(batch)))
        x_real.append(torch.cat([_x_real, grayscale(_x_real)], 0))
        t_img.append(torch.cat([_t_img, grayscale(_t_img)], 0))
        shape_mask.append(batch[i][2])
        mask.append(batch[i][3])
        label_org.append(batch[i][4].to(dtype=torch.int32))

    return torch.stack(x_real), \
           torch.stack(t_img), \
           torch.stack(shape_mask), \
           torch.stack(mask), \
           torch.LongTensor(label_org)


def train(args):
    glr = args.lr
    dlr = args.ttur
    set_seed(args.random_seed)
    if args.mode == "train":
        syn_dataset = ChaosDataset_Syn_new(path=args.dataset_path, split='train', modals=args.modals,
                                           image_size=args.image_size)
        syn_loader = DataLoader(syn_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=None )#if (
                # args.real or (not args.real and args.soup)) else convert_data_for_quaternion_tarGAN)
    ### eval during training
    syneval_dataset = ChaosDataset_Syn_Test(path=args.dataset_path, modal=args.modals[0], gan=True,
                                            image_size=args.image_size)
    syneval_dataset2 = ChaosDataset_Syn_Test(path=args.dataset_path, modal=args.modals[1], gan=True,
                                             image_size=args.image_size)
    syneval_dataset3 = ChaosDataset_Syn_Test(path=args.dataset_path, modal=args.modals[2], gan=True,
                                             image_size=args.image_size)
    # if args.mode =="eval":
    syneval_dataset4 = ChaosDataset_Syn_new(path=args.dataset_path, split='test', modals=args.modals,
                                            image_size=args.image_size)
    syneval_loader = DataLoader(syneval_dataset4, batch_size=args.batch_size,
                                shuffle=True if args.mode != "sample" else False, collate_fn=None )#if (
                # args.real or (not args.real and args.soup)) else convert_data_for_quaternion_tarGAN)

    nets, disc_c_dim = build_model()
    optims = build_optims(nets, glr, dlr)

    tot = 0
    for name, module in nets.items():
        if name != "netG_use":
            tot += print_network(module, name)

    print("DONE", tot)
    if args.sepoch > 0:
        load_nets(nets)

    start_time = time.time()
    print('start training...')
    ii = 0  # 22127 #25 epoch =
    with wandb.init(config=args, project="quattargan") as run:
        wandb.run.name = args.experiment_name
        for i in tqdm(range(args.sepoch, args.epoch), initial=args.sepoch, total=args.epoch):
            for epoch, (x_real, t_img, shape_mask, mask, label_org) in tqdm(enumerate(syn_loader),
                                                                            total=len(syn_loader),
                                                                            desc="epoch {}".format(i)):
                # 1. Preprocess input data
                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]
                c_org = label2onehot(label_org, args.c_dim)
                c_trg = label2onehot(label_trg, args.c_dim)
                d_false_org = label2onehot(label_org + args.c_dim, disc_c_dim-2)
                d_org = label2onehot(label_org, disc_c_dim-2)
                g_trg = label2onehot(label_trg, disc_c_dim-2)
                x_real = x_real.to(device)  # Input images.
                c_org = c_org.to(device)  # Original domain labels.
                c_trg = c_trg.to(device)  # Target area domain labels y.
                d_org = d_org.to(device)  # Labels for computing classification loss.
                g_trg = g_trg.to(device)  # Labels for computing classification loss.
                d_false_org = d_false_org.to(device)  # Labels for computing classification loss.
                mask = mask.to(device)
                shape_mask = shape_mask.to(device)
                t_img = t_img.to(device)
                index = loss_filter(mask, device)
                # 2. Train the discriminator
                # Compute loss with real whole images.
                out_src, out_cls = nets.netD_i(x_real)
                # print("out src out cls ",out_src.shape,out_cls.shape)
                # print("out src",out_src.shape,out_cls.shape)
                d_loss_real = -torch.mean(out_src)
                if out_cls.size(1) != d_org.size(1):
                    out_cls = out_cls[:,:d_org.size(1)]
                d_loss_cls = F.binary_cross_entropy_with_logits(out_cls, d_org, reduction='sum') / out_cls.size(0)
                # Compute loss with fake whole images.
                with torch.no_grad():
                    x_fake, t_fake = nets.netG(x_real, t_img,
                                               c_trg)  # G(image,target_image,target_modality) --> (out_image,output_target_area_image)
                    # print("out x fake t fake", x_fake.shape,t_fake.shape)
                out_src, out_f_cls = nets.netD_i(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # wavelet
                #out_f_cls = torch.cat([out_f_cls, torch.zeros(out_f_cls.size(0), 4)], dim=1)
                if out_f_cls.size(1) != d_false_org.size(1):
                    out_f_cls = out_f_cls[:,:d_false_org.size(1)]
                d_loss_f_cls = F.binary_cross_entropy_with_logits(out_f_cls, d_false_org,
                                                                  reduction='sum') / out_f_cls.size(
                    0)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(device)
                x_hat = (alpha * x_real[:, 1:].data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = nets.netD_i(x_hat)
                d_loss_gp = gradient_penalty(out_src, x_hat, device)

                # compute loss with target images
                if index.shape[0] != 0:
                    out_src, out_cls = nets.netD_t(torch.index_select(t_img, dim=0, index=index))
                    # print("Target out src out cls ",out_src.shape,out_cls.shape)

                    d_org = torch.index_select(d_org, dim=0, index=index)
                    d_loss_real_t = -torch.mean(out_src)
                    if out_cls.size(1) != d_org.size(1):
                        out_cls = out_cls[:, :d_org.size(1)]
                    d_loss_cls_t = F.binary_cross_entropy_with_logits(out_cls, d_org, reduction='sum') / out_cls.size(0)

                    out_src, out_f_cls = nets.netD_t(torch.index_select(t_fake.detach(), dim=0, index=index))
                    #wavelet
                    if out_f_cls.size(1) != d_false_org.size(1):
                        out_f_cls = out_f_cls[:, :d_false_org.size(1)]
                    d_false_org = torch.index_select(d_false_org, dim=0, index=index)
                    d_loss_fake_t = torch.mean(out_src)
                    d_loss_f_cls_t = F.binary_cross_entropy_with_logits(out_f_cls, d_false_org,
                                                                        reduction='sum') / out_f_cls.size(0)

                    x_hat = (alpha * t_img[:, 1:].data + (1 - alpha) * t_fake.data).requires_grad_(True)
                    x_hat = torch.index_select(x_hat, dim=0, index=index)
                    out_src, _ = nets.netD_t(x_hat)
                    d_loss_gp_t = gradient_penalty(out_src, x_hat, device)

                    dt_loss = d_loss_real_t + d_loss_fake_t + d_loss_cls_t + d_loss_gp_t * 10 + d_loss_f_cls_t * args.w_d_false_t_c
                    w_dt = (-d_loss_real_t - d_loss_fake_t).item()
                else:
                    dt_loss = torch.FloatTensor([0]).to(device)
                    w_dt = 0
                    d_loss_f_cls_t = torch.FloatTensor([0]).to(device)
                # Backward and optimize.
                di_loss = d_loss_real + d_loss_fake + d_loss_cls + d_loss_gp * 10 + d_loss_f_cls * args.w_d_false_c
                d_loss = di_loss + dt_loss
                w_di = (-d_loss_real - d_loss_fake).item()

                optims.g_optimizier.zero_grad()
                optims.di_optimizier.zero_grad()
                optims.dt_optimizier.zero_grad()
                d_loss.backward()
                optims.di_optimizier.step()
                optims.dt_optimizier.step()

                #  3. Train the generator
                # Original-to-target domain.
                x_fake, t_fake = nets.netG(x_real, t_img, c_trg)
                out_src, out_cls = nets.netD_i(x_fake)
                g_loss_fake = -torch.mean(out_src)
                if out_cls.size(1) != g_trg.size(1):
                    out_cls = out_cls[:, :g_trg.size(1)]
                g_loss_cls = F.binary_cross_entropy_with_logits(out_cls, g_trg, reduction='sum') / out_cls.size(0)
                # print("shape shape",netH(x_fake).shape)

                shape_loss = F.mse_loss(nets.netH(x_fake), shape_mask.float())
                # Target-to-original domain.
                x_reconst, t_reconst = nets.netG(x_fake, t_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                if index.shape[0] != 0:
                    out_src, out_cls = nets.netD_t(torch.index_select(t_fake, dim=0, index=index))
                    g_trg = torch.index_select(g_trg, dim=0, index=index)
                    g_loss_fake_t = -torch.mean(out_src)
                    if out_cls.size(1) != g_trg.size(1):
                        out_cls = out_cls[:, :g_trg.size(1)]
                    g_loss_cls_t = F.binary_cross_entropy_with_logits(out_cls, g_trg, reduction='sum') / out_cls.size(0)
                    gt_loss = g_loss_fake_t + g_loss_cls_t * args.w_g_t_c
                else:
                    gt_loss = torch.FloatTensor([0]).to(device)
                    g_loss_cls_t = torch.FloatTensor([0]).to(device)

                shape_loss_t = F.mse_loss(nets.netH(t_fake), mask.float())
                g_loss_rec_t = torch.mean(torch.abs(t_img - t_reconst))
                cross_loss = torch.mean(torch.abs(denorm(x_fake) * mask - denorm(t_fake)))
                # Backward and optimize.
                gi_loss = g_loss_fake + args.w_cycle * g_loss_rec + g_loss_cls * args.w_g_c + shape_loss * args.w_shape
                gt_loss = gt_loss + args.w_cycle * g_loss_rec_t + shape_loss_t * args.w_shape + cross_loss * args.w_g_cross
                g_loss = gi_loss + gt_loss

                optims.g_optimizier.zero_grad()
                optims.di_optimizier.zero_grad()
                optims.dt_optimizier.zero_grad()
                optims.h_optimizier.zero_grad()
                g_loss.backward()
                optims.g_optimizier.step()
                optims.h_optimizier.step()

                moving_average(nets.netG, nets.netG_use, beta=0.999)

                if (epoch + 0) % args.log_every == 0:
                    all_losses = dict()

                    all_losses["train/D/w_di"] = w_di
                    all_losses["train/D/w_dt"] = w_dt
                    all_losses["train/D/loss_f_cls"] = d_loss_f_cls.item()
                    all_losses["train/D/loss_f_cls_t"] = d_loss_f_cls_t.item()
                    all_losses["train/G/loss_cls"] = g_loss_cls.item()
                    all_losses["train/G/loss_cls_t"] = g_loss_cls_t.item()
                    all_losses["train/G/loss_shape"] = shape_loss.item()
                    all_losses["train/G/loss_shape_t"] = shape_loss_t.item()
                    all_losses["train/G/loss_cross"] = cross_loss.item()
                    wandb.log(all_losses, step=ii, commit=True)

                    # vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)

                ii = ii + 1
                ###################################

            # show syn images after every epoch

            if (i + 1) % 1 == 0 and (i + 1) > 0:
                # if (epoch + 0) % args.print_every == 0:
                x_concat = plot_images(nets.netG_use, i, syneval_dataset, syneval_dataset2, syneval_dataset3)
                wandb.log({"sample_dir": wandb.Image(x_concat, caption="epoch " + str(i))}, commit=False)
                del x_concat

            if (i + 1) % args.save_every == 0:
                args.net_name = 'netG'
                save_state_net(nets.netG, args, i + 1, optims.g_optimizier)
                args.net_name = 'netG_use'
                save_state_net(nets.netG_use, args, i + 1, None)
                args.net_name = 'netDi'
                save_state_net(nets.netD_i, args, i + 1, optims.di_optimizier)
                args.net_name = 'netDt'
                save_state_net(nets.netD_t, args, i + 1, optims.dt_optimizier)
                args.net_name = 'netH'
                save_state_net(nets.netH, args, i + 1, optims.h_optimizier)

            if (i + 1) == args.epoch:
                fidstar, fid, dice, ravd, s_score, fid_giov = calculate_all_metrics()
                wandb.log(dict(fidstar), step=ii + 1, commit=False)
                wandb.log(dict(fid), step=ii + 1, commit=False)
                wandb.log(dict(fid_giov), step=ii + 1, commit=False)

                wandb.log(dict(dice), step=ii + 1, commit=False)
                wandb.log(dict(ravd), step=ii + 1, commit=False)
                wandb.log(dict(s_score), step=ii + 1, commit=True)
            if (i + 1) % 1 == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i + 1, args.epoch)
                print(log)
                torch.cuda.empty_cache()


def build_model():
    if not args.real and args.soup:
        disc_c_dim = 8
    elif args.real:
        disc_c_dim = args.c_dim * 2
    else:
        #wavelets
        disc_c_dim = 8
    channels = 5 if (not args.real and not args.soup) else 1

    netG = Generator(in_c=channels + args.c_dim, mid_c=args.G_conv, layers=2, s_layers=3, affine=True, last_ac=True).to(
        device)

    shape_net_channels = 4 if not args.real and not args.last_layer_gen_real else 1
    netH = ShapeUNet(img_ch=shape_net_channels, mid=args.h_conv, output_ch=shape_net_channels).to(device)

    netD_i = Discriminator(c_dim=disc_c_dim, image_size=args.image_size).to(device)
    netD_t = Discriminator(c_dim=disc_c_dim, image_size=args.image_size).to(device)

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
                        "netG_use": netG_use})
    return nets, disc_c_dim


def build_optims(nets, glr, dlr):
    g_optimizier = torch.optim.Adam(nets.netG.parameters(), lr=glr, betas=(args.betas[0], args.betas[1]))
    di_optimizier = torch.optim.Adam(nets.netD_i.parameters(), lr=dlr, betas=(args.betas[0], args.betas[1]))
    dt_optimizier = torch.optim.Adam(nets.netD_t.parameters(), lr=dlr, betas=(args.betas[0], args.betas[1]))
    h_optimizier = torch.optim.Adam(nets.netH.parameters(), lr=glr, betas=(args.betas[0], args.betas[1]))

    return munch.Munch({"g_optimizier": g_optimizier, "di_optimizier": di_optimizier, "dt_optimizier": dt_optimizier,
                        "h_optimizier": h_optimizier})


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))
    return num_params


def load_nets(nets):
    for net in nets.keys():
        print("loading", net)
        net_check = net if "use" in net else net.replace("_", "")
        load_state_net(nets[net], net_check, args.sepoch)
