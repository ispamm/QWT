import os
import time
from importlib import reload
import dataset, train, utils
reload(dataset)
reload(train)
reload(utils)
import configs.config_tmp
reload(configs.config_tmp)
from datetime import  timedelta
from dataset import ChaosDataset_Syn_new
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config_tmp import args, device
from train import build_model, build_optims, load_nets
from utils import label2onehot, getLabel, save_image


def sample(experiment=["parcollet_nuoovo", "phm_nuoovo"],):
    mod = ["t1", "t2", "ct"]
    syneval_dataset4 = ChaosDataset_Syn_new(path=args.dataset_path, split='test', modals=args.modals,
                                            image_size=args.image_size)
    syneval_loader = DataLoader(syneval_dataset4, batch_size=args.eval_batch_size,
                                shuffle=False, collate_fn=None)
    for exp in tqdm(experiment):

        start_time = time.time()
        nets, _ = build_model()
        #optims = build_optims(nets)
        load_nets(nets)
        print(exp)
        # os.makedirs("results/translation/"+exp)
        # os.makedirs("results/segmentation/"+exp)
        # os.makedirs("results/groundTrans/"+exp)
        # os.makedirs("results/groundSeg/"+exp)
        for idx_eval in tqdm(range(3)):
            # loaders = {
            #     "t1_loader": DataLoader(syneval_dataset, batch_size=args.eval_batch_size),
            #     "t2_loader": DataLoader(syneval_dataset2, batch_size=args.eval_batch_size),
            #     "ct_loader": DataLoader(syneval_dataset3, batch_size=args.eval_batch_size)
            # }
            for epoch, ((x_real, wavelet_real), (t_img, wavelet_target), shape_mask, mask, label_org) in tqdm(enumerate(syneval_loader),
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
                    filename = os.path.join("results/translation",
                                            mod[idx_eval] + "_" + exp + '_%.4i_%.2i.png' % (
                                                args.sepoch * args.eval_batch_size + (k + 1), epoch + 1))
                    save_image(x_fake[k], ncol=1, filename=filename)
                    filename = os.path.join("results/groundTrans",
                                            mod[idx_eval] + "_" + exp + '_%.4i_%.2i.png' % (
                                                args.sepoch * args.eval_batch_size + (k + 1), epoch + 1))
                    save_image(x_real[k], ncol=1, filename=filename)
                    filename = os.path.join("results/segmentation",
                                            mod[idx_eval] + "_" + exp + '_%.4i_%.2i.png' % (
                                                args.sepoch * args.eval_batch_size + (k + 1), epoch + 1))
                    save_image(t_fake[k], ncol=1, filename=filename)
                    filename = os.path.join("results/groundSeg",
                                            mod[idx_eval] + "_" + exp + '_%.4i_%.2i.png' % (
                                                args.sepoch * args.eval_batch_size + (k + 1), epoch + 1))
                    save_image(t_img[k], ncol=1, filename=filename)
                # To calculate inference time
                # if epoch==1:
                #     break

        elapsed = time.time() - start_time
        elapsed = str(timedelta(seconds=elapsed))[:-7]
        log = "Elapsed time [%s], " % (elapsed)
        print(args.experiment_name, log)
