import os
import time
# from importlib import reload
# import dataset, train, utils
# reload(dataset)
# reload(train)
# reload(utils)
# import configs.config_tmp
# reload(configs.config_tmp)
from datetime import  timedelta
from dataset import ChaosDataset_Syn_new
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import args, device
from train import build_model, build_optims, load_nets
from utils import label2onehot, getLabel, save_image


def sample(nets=None, experiment=["parcollet_nuoovo", "phm_nuoovo"],):
    mod = ["t1", "t2", "ct"]
    syneval_dataset4 = ChaosDataset_Syn_new(path=args.dataset_path, split='test', modals=args.modals,
                                            image_size=args.image_size)
    syneval_loader = DataLoader(syneval_dataset4, batch_size=args.eval_batch_size,
                                shuffle=False, collate_fn=None)
    for exp in tqdm(experiment):

        start_time = time.time()
        # nets, _ = build_model()
        # #optims = build_optims(nets)
        # load_nets(nets)
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


mappa = {'targan_1245135' : "TarGAN", 'wtargan_1245135':"DWT", 'qwtargan_best4_1245135':"QWT-SGS", 'qwtargan_best4_moe_1245135':"QWT-MoE",}
#qwtargan_best4_1245135
def paper_sampling(randint,randint2, nets_names=['targan_1245135', 'wtargan_1245135', 'qwtargan_best4_moe_1245135']):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    plt.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(15, 5))

    plt.axis('off')
    plt.subplot(3,9,1)
    plt.imshow(mpimg.imread("/home/luigi/Documents/QWT/TarGAN/results/groundTrans/ct_qwtargan_best4_moe_1245135_000"+randint[0]+"_"+randint2[0]+".png"))
    plt.xticks([]) 
    plt.yticks([])
    plt.ylabel(r"CT $\rightarrow$ T1, T2")

    j=0
    for i,net_name in enumerate(nets_names):
        plt.subplot(3,9,i+2+j)
        img = mpimg.imread('/home/luigi/Documents/QWT/TarGAN/results/translation/t1_'+net_name+'_000'+randint[0]+"_"+randint2[0]+'.png')
        plt.imshow(img)
        plt.xticks([]) 
        plt.yticks([])
        plt.subplot(3,9,i+3+j)
        img = mpimg.imread('/home/luigi/Documents/QWT/TarGAN/results/translation/t2_'+net_name+'_000'+randint[0]+"_"+randint2[0]+'.png')
        plt.imshow(img)        #plt.xlabel(net_name+str(i))
        plt.xticks([]) 
        plt.yticks([])
        j+=1
        plt.subplots_adjust(wspace=0, hspace=0)

    plt.subplot(3,9,10)
    plt.imshow(mpimg.imread("/home/luigi/Documents/QWT/TarGAN/results/groundTrans/t1_qwtargan_best4_moe_1245135_000"+randint[1]+"_"+randint2[1]+".png"))
    plt.xticks([]) 
    plt.yticks([])
    plt.ylabel(r"T1 $\rightarrow$ T2, CT")

    j=9
    for i,net_name in enumerate(nets_names):
        plt.subplot(3,9,i+2+j)
        img = mpimg.imread('/home/luigi/Documents/QWT/TarGAN/results/translation/t2_'+net_name+'_000'+randint[1]+"_"+randint2[1]+'.png')
        plt.imshow(img)
        plt.xticks([]) 
        plt.yticks([])
        plt.subplot(3,9,i+3+j)
        img = mpimg.imread('/home/luigi/Documents/QWT/TarGAN/results/translation/ct_'+net_name+'_000'+randint[1]+"_"+randint2[1]+'.png')
        plt.imshow(img)
        plt.xticks([]) 
        plt.yticks([])
        #plt.xlabel(net_name+str(i))
        j+=1
        plt.subplots_adjust(wspace=0, hspace=0)

    plt.subplot(3,9,19)
    plt.imshow(mpimg.imread("/home/luigi/Documents/QWT/TarGAN/results/groundTrans/t2_qwtargan_best4_moe_1245135_000"+randint[2]+"_"+randint2[2]+".png"))
    plt.xticks([]) 
    plt.yticks([])
    plt.ylabel(r"T2 $\rightarrow$ T1, CT")

    #plt.xlabel("Source")

    j=18
    for i,net_name in enumerate(nets_names):
        plt.subplot(3,9,i+2+j)
        img = mpimg.imread('/home/luigi/Documents/QWT/TarGAN/results/translation/t1_'+net_name+'_000'+randint[2]+"_"+randint2[2]+'.png')
        plt.imshow(img)
        plt.xticks([]) 
        plt.yticks([])
        plt.subplot(3,9,i+3+j)
        img = mpimg.imread('/home/luigi/Documents/QWT/TarGAN/results/translation/ct_'+net_name+'_000'+randint[2]+"_"+randint2[2]+'.png')
        plt.imshow(img)
        plt.xticks([]) 
        plt.yticks([])
        #plt.xlabel(mappa[net_name])
        j+=1

    # plt.subplot(2,2,2)
    # plt.imshow(imgs[i])
    # plt.subplot(2,2,3)
    # plt.imshow(imgs[i])
    # plt.subplots_adjust(wspace=0.05, hspace=0)

    plt.text(-710, 145, 'Source', ha='center')
    plt.text(-525, 145, 'TarGAN', ha='center')
    plt.text(-270, 145, 'TarGAN + DWT', ha='center')
    # plt.text(-260, 145, 'TarGAN + QWT-SGS', ha='center')
    # plt.text(10, 145, 'TarGAN + QWT-MoE', ha='center')
    plt.text(10, 145, 'TarGAN + QWT-SGS', ha='center')
    line = plt.Line2D((.213,.213),(.1,.9), color="w", linewidth=2)
    fig.add_artist(line)

    line = plt.Line2D((.385,.385),(.1,.9), color="w", linewidth=2)
    fig.add_artist(line)

    line = plt.Line2D((.56,.56),(.1,.9), color="w", linewidth=2)
    fig.add_artist(line)
    
    # line = plt.Line2D((.73,.73),(.1,.9), color="w", linewidth=2)
    # fig.add_artist(line)
    plt.savefig("/home/luigi/Documents/QWT/TarGAN/results/targan_results_"+randint2[0]+randint[1]+".png", dpi=300, bbox_inches="tight")
    print("targan_results_"+randint2[0]+randint[1], "\n")
    print(randint, "\n", randint2)
    print("----------------------------")


import random
for i in range(20):
    # paper_sampling([str(random.randint(1,7)).zfill(1), str(random.randint(1,7)).zfill(1), str(random.randint(1,7)).zfill(1)], 
    #                 [str(random.randint(23,30)).zfill(2),str(random.randint(1,10)).zfill(2),str(random.randint(10,20)).zfill(2)])

    paper_sampling(
    ['3', '1', '1'], 
 ['30', '08', '16'])