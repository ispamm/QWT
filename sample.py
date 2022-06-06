import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import args, device
from train import build_model, build_optims, load_nets
from utils import label2onehot, getLabel, save_image


def sample(
        syneval_dataset,
        syneval_dataset2,
        syneval_dataset3,
        syneval_loader,
        experiment=["parcollet_nuoovo", "phm_nuoovo"],):
    mod = ["t1", "t2", "ct"]
    for exp in tqdm(experiment):
        args.experiment_name = exp
        if "phm" in exp:
            args.qsn = False
            args.phm = True
            args.real = False
            args.soup = True
        elif "parc" in exp:
            args.qsn = True
            args.phm = False
            args.real = False
            args.soup = True
        elif "real" in exp:
            args.qsn = False
            args.phm = False
            args.real = True
            args.soup = False
        if "nuo" in exp:
            args.last_layer_gen_real = True
            args.share_net_real = True
        else:
            args.last_layer_gen_real = False
            args.share_net_real = False
        start_time = time.time()

        nets, _ = build_model()
        optims = build_optims(nets)
        load_nets(nets)
        print(exp)

        for idx_eval in tqdm(range(3)):
            loaders = {
                "t1_loader": DataLoader(syneval_dataset, batch_size=args.eval_batch_size),
                "t2_loader": DataLoader(syneval_dataset2, batch_size=args.eval_batch_size),
                "ct_loader": DataLoader(syneval_dataset3, batch_size=args.eval_batch_size)
            }
            for epoch, (x_real, t_img, shape_mask, mask, label_org) in tqdm(enumerate(syneval_loader),
                                                                            total=len(syneval_loader)):
                rand_idx = torch.randperm(label_org.size(0))
                # label_trg = label_org[rand_idx]
                c_org = label2onehot(label_org, args.c_dim)
                # c_trg = label2onehot(label_trg, args.c_dim)
                x_real = x_real.to(device)  # Input images.
                c_org = c_org.to(device)  # Original domain labels.
                # c_trg = c_trg.to(device)
                t_img = t_img.to(device)
                # translate only in one domain
                # if dice_:
                # s = c_trg.size(0)
                # c_trg = c_trg[:s]
                c_trg = getLabel(x_real, device, idx_eval, args.c_dim)

                # Original-to-target domain.

                # if not dice_:
                #     c_t,x_r,t_i,c_o = [],[],[],[]
                #     for i,x in enumerate(c_trg):
                #         if not torch.all(x.eq(c_org[i])):
                #             c_t.append(x)
                #             x_r.append(x_real[i])
                #             t_i.append(t_img[i])
                #             c_o.append(c_org[i])

                #         # print(x,c_org[i])
                #     if len(c_t) == 0:
                #         continue
                #     c_trg = torch.stack(c_t,dim=0).to(device)
                #     x_real = torch.stack(x_r,dim=0).to(device)
                #     t_img = torch.stack(t_i,dim=0).to(device)
                #     c_org = torch.stack(c_o,dim=0).to(device)

                # good for dice
                x_fake, t_fake = nets.netG_use(x_real, t_img,
                                               c_trg)  # G(image,target_image,target_modality) --> (out_image,output_target_area_image)

                # if not dice_:
                #     x_reconst, t_reconst = netG(x_fake, t_fake, c_org)
                #     t_fake = t_reconst

                for k in range(c_trg.size(0)):
                    filename = os.path.join("/content/drive/MyDrive/Thesis/TarGAN/results/translation",
                                            mod[idx_eval] + "_" + exp + '_%.4i_%.2i.png' % (
                                                args.sepoch * args.eval_batch_size + (k + 1), epoch + 1))
                    save_image(x_fake[k], ncol=1, filename=filename)
                    filename = os.path.join("/content/drive/MyDrive/Thesis/TarGAN/results/groundTrans",
                                            mod[idx_eval] + "_" + exp + '_%.4i_%.2i.png' % (
                                                args.sepoch * args.eval_batch_size + (k + 1), epoch + 1))
                    save_image(x_real[k], ncol=1, filename=filename)
                    filename = os.path.join("/content/drive/MyDrive/Thesis/TarGAN/results/segmentation",
                                            mod[idx_eval] + "_" + exp + '_%.4i_%.2i.png' % (
                                                args.sepoch * args.eval_batch_size + (k + 1), epoch + 1))
                    save_image(t_fake[k], ncol=1, filename=filename)
                    filename = os.path.join("/content/drive/MyDrive/Thesis/TarGAN/results/groundSeg",
                                            mod[idx_eval] + "_" + exp + '_%.4i_%.2i.png' % (
                                                args.sepoch * args.eval_batch_size + (k + 1), epoch + 1))
                    save_image(t_img[k], ncol=1, filename=filename)
                # To calculate inference time
                # if epoch==1:
                #     break

        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
        log = "Elapsed time [%s], " % (elapsed)
        print(args.experiment_name, log)
