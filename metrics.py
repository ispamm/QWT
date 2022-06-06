import os
import shutil
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import device, args
from pytorch import unet3d
from utils import save_image, getLabel, label2onehot
import pydicom
import numpy as np
import glob
import cv2
import SimpleITK as sitk
from scipy import ndimage
from sklearn.neighbors import KDTree


def calculate_fid():
    if "50" in args.png_dataset_path:
        path_real = [args.dataset_path+"png5050/train/ct",args.dataset_path+"png5050/train/t1",args.dataset_path+"png5050/train/t2"]
    else:
        path_real = [args.dataset_path+"png8020/train/ct",args.dataset_path+"png8020/train/t1",args.dataset_path+"png8020/train/t2"]

    eval_root = "/content/drive/MyDrive/Thesis/TarGAN/eval/"
    fid_scores = {}
    for p in path_real:
        mod = [m for m in args.modals if m!=p[-2:]]
        ls = 0
        for src in mod:
            print("evaluating " + src +" to "+ p[-2:])
            eval_path = eval_root +src +" to "+ p[-2:]
            x = os.system(f'python -m pytorch_fid "{p}" "{eval_path}" --device "cuda:0" --batch-size {args.eval_batch_size}')
            x = x[-1].replace("FID:  ","")
            fid_scores["FID/"+src +" to "+ p[-2:]] = float(x)
            ls += float(x)
        fid_scores["FID/"+p[-2:]+"_mean"] = ls/len(mod)
    return fid_scores

'''
Stargan v2 metrics
'''

@torch.no_grad()
def calculate_metrics(nets, args, step, mode):
    print('Calculating evaluation metrics...')
    domains = os.listdir(args.val_img_dir)
    domains.sort()
    num_domains = len(domains)
    #calculate_fid_for_all_tasks(args, domains, step=step, mode=mode)
    print('Number of domains: %d' % num_domains)
    lpips_dict = OrderedDict()
    loaders = {
        "t1_loader":DataLoader(syneval_dataset,batch_size=args.eval_batch_size),
        "t2_loader": DataLoader(syneval_dataset2,batch_size=args.eval_batch_size),
        "ct_loader":DataLoader(syneval_dataset3,batch_size=args.eval_batch_size)
    }
    mod = {"t1":0,"t2":1,"ct":2}

    #loaders = (syneval_dataset, syneval_dataset2,syneval_dataset3)
    #loaders = (DataLoader(syneval_dataset,batch_size=4), DataLoader(syneval_dataset2,batch_size=4),DataLoader(syneval_dataset3,batch_size=4))

    for trg_idx, trg_domain in enumerate(domains):
        src_domains = [x for x in domains if x != trg_domain]
        loader_ref = loaders[trg_domain+"_loader"]
        path_ref = os.path.join(args.png_dataset_path+'/eval', trg_domain)
        # loader_ref = get_eval_loader(root=path_ref,
        #                                  img_size=args.image_size,
        #                                  batch_size=args.eval_batch_size,
        #                                  imagenet_normalize=False,
        #                                  drop_last=True)
        for src_idx, src_domain in enumerate(src_domains):
            loader_src = loaders[src_domain+"_loader"]
            path_src = os.path.join(args.png_dataset_path+'/eval', src_domain)
            # loader_src = get_eval_loader(root=path_src,
            #                              img_size=args.image_size,
            #                              batch_size=args.eval_batch_size,
            #                              imagenet_normalize=False)
            task = '%s to %s' % (src_domain, trg_domain)
            path_fake = os.path.join(args.eval_dir, task)
            shutil.rmtree(path_fake, ignore_errors=True)
            os.makedirs(path_fake)
            lpips_values = []
            print('Generating images and calculating LPIPS for %s...' % task)
            for i, (x_src,x_msk) in enumerate(tqdm(loader_src, total=len(loader_src))):
                N = x_src.size(0)
                x_src = x_src.to(device)
                #y_trg = torch.tensor([trg_idx] * N).to(device)
                # generate 10 outputs from the same input
                group_of_images = []
                for j in range(10): #num outs per domain
                    try:
                        x_ref = next(iter_ref).to(device)
                    except:
                        iter_ref = iter(loader_ref)
                        x_ref,x_ref_msk = next(iter_ref)
                        x_ref = x_ref.to(device)

                    if x_ref.size(0) > N:
                        x_ref = x_ref[:N]
                    #idx = random.choice([0,1,2]) #parte da zoomare?
                    #x_src_batch = x_src.unsqueeze(0)
                    idx = mod[trg_domain]
                    c = getLabel(x_src, device, idx, args.c_dim)
                    x_src = x_src[:,:1,:,:]
                    x_fake = nets.netG_use(x_src, None, c, mode='test')
                    group_of_images.append(x_fake)
                    # save generated images to calculate FID later
                    for k in range(N):
                        filename = os.path.join(path_fake, '%.4i_%.2i.png' % (i*args.eval_batch_size+(k+1), j+1))
                        save_image(x_fake[k], ncol=1, filename=filename)

                lpips_value = calculate_lpips_given_images(group_of_images)
                lpips_values.append(lpips_value)
            #print(lpips_values)
            # calculate LPIPS for each task (e.g. cat2dog, dog2cat)
            lpips_mean = np.array(lpips_values).mean()
            lpips_dict['LPIPS_%s/%s' % (mode, task)] = lpips_mean

        # delete dataloaders
        del loader_src
        if mode == 'test':
            del loader_ref
            del iter_ref

    # calculate the average LPIPS for all tasks
    lpips_mean = 0
    for _, value in lpips_dict.items():
        lpips_mean += value / len(lpips_dict)
    lpips_dict['LPIPS_%s/mean' % mode] = lpips_mean

    # report LPIPS values
    filename = os.path.join(args.eval_dir, 'LPIPS_%.5i_%s.json' % (step, mode))
    save_json(lpips_dict, filename)

    # calculate and report fid values
    return lpips_dict, calculate_fid_for_all_tasks(args, domains, step=step, mode=mode)

def normalize_lpips(x, eps=1e-10):
    return x * torch.rsqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = models.alexnet(pretrained=True).features
        self.channels = []
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                self.channels.append(layer.out_channels)

    def forward(self, x):
        fmaps = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                fmaps.append(x)
        return fmaps


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))

    def forward(self, x):
        return self.main(x)


class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.alexnet = AlexNet()
        self.lpips_weights = nn.ModuleList()
        for channels in self.alexnet.channels:
            self.lpips_weights.append(Conv1x1(channels, 1))
        self._load_lpips_weights()
        # imagenet normalization for range [-1, 1]
        self.mu = torch.tensor([-0.03, -0.088, -0.188]).view(1, 3, 1, 1).to(device)
        self.sigma = torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1).to(device)

    def _load_lpips_weights(self):
        own_state_dict = self.state_dict()
        if torch.cuda.is_available():
            state_dict = torch.load(args.checkpoint_dir+'/lpips_weights.ckpt')
        else:
            state_dict = torch.load(args.checkpoint_dir+'/lpips_weights.ckpt',
                                    map_location=torch.device('cpu'))
        for name, param in state_dict.items():
            if name in own_state_dict:
                own_state_dict[name].copy_(param)

    def forward(self, x, y):
        x = (x - self.mu) / self.sigma
        y = (y - self.mu) / self.sigma
        x_fmaps = self.alexnet(x)
        y_fmaps = self.alexnet(y)
        lpips_value = 0
        for x_fmap, y_fmap, conv1x1 in zip(x_fmaps, y_fmaps, self.lpips_weights):
            x_fmap = normalize_lpips(x_fmap)
            y_fmap = normalize_lpips(y_fmap)
            lpips_value += torch.mean(conv1x1((x_fmap - y_fmap)**2))
        return lpips_value


@torch.no_grad()
def calculate_lpips_given_images(group_of_images):
    # group_of_images = [torch.randn(N, C, H, W) for _ in range(10)]
    lpips = LPIPS().eval().to(device)
    lpips_values = []
    num_rand_outputs = len(group_of_images)

    # calculate the average of pairwise distances among all random outputs
    for i in range(num_rand_outputs-1):
        for j in range(i+1, num_rand_outputs):
            lpips_values.append(lpips(group_of_images[i].repeat(1,3, 1, 1), group_of_images[j].repeat(1,3, 1, 1)))

    lpips_value = torch.mean(torch.stack(lpips_values, dim=0))
    return lpips_value.item()

'''
FID
'''

def calculate_fid_for_all_tasks(args, domains, step, mode):
    print('Calculating FID for all tasks...')
    fid_values = OrderedDict()
    for trg_domain in domains:
        src_domains = [x for x in domains if x != trg_domain]

        for src_domain in src_domains:
            task = '%s to %s' % (src_domain, trg_domain)
            path_real = os.path.join(args.dataset_path+"train", trg_domain)
            path_fake = os.path.join(args.eval_dir, task)
            print('Calculating FID for %s...' % task)
            fid_value = calculate_fid_given_paths(
                paths=[path_real, path_fake],
                img_size=args.image_size,
                batch_size=args.eval_batch_size
                )
            fid_values['FID_%s/%s' % (mode, task)] = fid_value

    # calculate the average FID for all tasks
    fid_mean = 0
    for _, value in fid_values.items():
        fid_mean += value / len(fid_values)
    fid_values['FID_%s/mean' % mode] = fid_mean

    # report FID values
    filename = os.path.join(args.eval_dir, 'FID_%.5i_%s.json' % (step, mode))
    save_json(fid_values, filename)
    return fid_values


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    return np.real(dist)


@torch.no_grad()
def calculate_fid_given_paths(paths, img_size=256, batch_size=32):
    print('Calculating FID given paths %s and %s...' % (paths[0], paths[1]))
    inception = InceptionV3().eval().to(device)
    loaders = [get_eval_loader(path, img_size, batch_size) for path in paths]
    print(paths)
    mu, cov = [], []
    for i,loader in enumerate(loaders):
        actvs = []
        #print(paths[i])
        for x in tqdm(loader, total=len(loader)):
            try:
                sz = x.size(1)
                if sz == 1:
                    actv = inception(x.repeat(1,3, 1, 1).to(device))
                elif sz == 3:
                    actv = inception(x.to(device))
                else:
                    raise Exception("check FID dim")
            except:
                sz = x[0].size(1)
                if sz == 1:
                    actv = inception(x[0].repeat(1,3, 1, 1).to(device))
                elif sz == 3:
                    actv = inception(x[0].to(device))
                else:
                    raise Exception("check FID dim")

            actvs.append(actv)
        actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
        mu.append(np.mean(actvs, axis=0))
        cov.append(np.cov(actvs, rowvar=False))
    fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
    return fid_value

def get_eval_loader(path,image_size,batch_size):
    if "train" in path:
        #return DataLoader(MyDataset(args.dataset_path+"png/"+path[-2:]),batch_size=batch_size)
        return DataLoader(ChaosDataset_Syn_Test(path=path[:-9], modal=path[-2:], split='train',gan=True,image_size=image_size),batch_size=batch_size)
    else:
        return DataLoader(MyDataset(path),batch_size=batch_size)
        #return ChaosDataset_Syn_Test(path=path[:-13],modal=path[-8:], split="eval",gan=True,image_size=image_size)


# -*- coding: utf-8 -*-
"""
Created on 09/07/2019
@author: Ali Emre Kavur
"""


def evaluate(Vref, Vseg, dicom_dir):
    dice = DICE(Vref, Vseg)
    ravd = RAVD(Vref, Vseg)
    # [assd, mssd]=SSD(Vref,Vseg,dicom_dir)
    return dice, ravd  # , assd ,mssd


def DICE(Vref, Vseg):
    dice = 2 * (Vref & Vseg).sum() / (Vref.sum() + Vseg.sum()) * 100
    return dice


def RAVD(Vref, Vseg):
    ravd = (abs(Vref.sum() - Vseg.sum()) / Vref.sum()) * 100
    return ravd


def SSD(Vref, Vseg, dicom_dir):
    struct = ndimage.generate_binary_structure(3, 1)

    ref_border = Vref ^ ndimage.binary_erosion(Vref, structure=struct, border_value=1)
    ref_border_voxels = np.array(np.where(ref_border))

    seg_border = Vseg ^ ndimage.binary_erosion(Vseg, structure=struct, border_value=1)
    seg_border_voxels = np.array(np.where(seg_border))

    ref_border_voxels_real = transformToRealCoordinates(ref_border_voxels, dicom_dir)
    seg_border_voxels_real = transformToRealCoordinates(seg_border_voxels, dicom_dir)

    tree_ref = KDTree(np.array(ref_border_voxels_real))
    dist_seg_to_ref, ind = tree_ref.query(seg_border_voxels_real)
    tree_seg = KDTree(np.array(seg_border_voxels_real))
    dist_ref_to_seg, ind2 = tree_seg.query(ref_border_voxels_real)

    assd = (dist_seg_to_ref.sum() + dist_ref_to_seg.sum()) / (len(dist_seg_to_ref) + len(dist_ref_to_seg))
    mssd = np.concatenate((dist_seg_to_ref, dist_ref_to_seg)).max()
    return assd, mssd


def transformToRealCoordinates(indexPoints, dicom_dir):
    """
    This function transforms index points to the real world coordinates
    according to DICOM Patient-Based Coordinate System
    The source: DICOM PS3.3 2019a - Information Object Definitions page 499.

    In CHAOS challenge the orientation of the slices is determined by order
    of image names NOT by position tags in DICOM files. If you need to use
    real orientation data mentioned in DICOM, you may consider to use
    TransformIndexToPhysicalPoint() function from SimpleITK library.
    """

    dicom_file_list = glob.glob(dicom_dir + '/*.dcm')
    dicom_file_list.sort()
    # Read position and orientation info from first image
    ds_first = pydicom.dcmread(dicom_file_list[0])
    img_pos_first = list(map(float, list(ds_first.ImagePositionPatient)))
    img_or = list(map(float, list(ds_first.ImageOrientationPatient)))
    pix_space = list(map(float, list(ds_first.PixelSpacing)))
    # Read position info from first image from last image
    ds_last = pydicom.dcmread(dicom_file_list[-1])
    img_pos_last = list(map(float, list(ds_last.ImagePositionPatient)))

    T1 = img_pos_first
    TN = img_pos_last
    X = img_or[:3]
    Y = img_or[3:]
    deltaI = pix_space[0]
    deltaJ = pix_space[1]
    N = len(dicom_file_list)
    M = np.array([[X[0] * deltaI, Y[0] * deltaJ, (T1[0] - TN[0]) / (1 - N), T1[0]],
                  [X[1] * deltaI, Y[1] * deltaJ, (T1[1] - TN[1]) / (1 - N), T1[1]],
                  [X[2] * deltaI, Y[2] * deltaJ, (T1[2] - TN[2]) / (1 - N), T1[2]], [0, 0, 0, 1]])

    realPoints = []
    for i in range(len(indexPoints[0])):
        P = np.array([indexPoints[1, i], indexPoints[2, i], indexPoints[0, i], 1])
        R = np.matmul(M, P)
        realPoints.append(R[0:3])

    return realPoints


def png_series_reader(dir):
    V = []
    png_file_list = glob.glob(dir + '/*.png')
    png_file_list.sort()
    for filename in png_file_list:
        image = cv2.imread(filename, 0)
        V.append(image)
    V = np.array(V, order='A')
    V = V.astype(bool)
    return V


def eval_dice_or_s_score(idx_eval, dice_=False):
    shutil.rmtree("Segmentation", ignore_errors=True)
    shutil.rmtree("Ground", ignore_errors=True)
    os.makedirs("Segmentation")
    os.makedirs("Ground")
    plotted = 0
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

        if not dice_:
            c_t, x_r, t_i, c_o = [], [], [], []
            for i, x in enumerate(c_trg):
                if not torch.all(x.eq(c_org[i])):
                    c_t.append(x)
                    x_r.append(x_real[i])
                    t_i.append(t_img[i])
                    c_o.append(c_org[i])

                # print(x,c_org[i])
            if len(c_t) == 0:
                continue
            c_trg = torch.stack(c_t, dim=0).to(device)
            x_real = torch.stack(x_r, dim=0).to(device)
            t_img = torch.stack(t_i, dim=0).to(device)
            c_org = torch.stack(c_o, dim=0).to(device)

        # good for dice
        x_fake, t_fake = nets.netG_use(x_real, t_img,
                                       c_trg)  # G(image,target_image,target_modality) --> (out_image,output_target_area_image)

        if not dice_:
            x_reconst, t_reconst = nets.netG(x_fake, t_fake, c_org)
            t_fake = t_reconst
        # Target-to-original domain.
        # fig = plt.figure(dpi=120)
        # with torch.no_grad():
        #     if plotted == 0:
        #         plt.subplot(241)
        #         plt.imshow(denorm(x_real[0]).squeeze().cpu().numpy(), cmap='gray')
        #         plt.title("original image")
        #         plt.subplot(242)
        #         plt.imshow(denorm(x_fake[0]).squeeze().cpu().numpy(), cmap='gray')
        #         plt.title("fake image")
        #         # plt.subplot(253)
        #         # plt.imshow(denorm(x_reconst[0]).squeeze().cpu().numpy(), cmap='gray')
        #         # plt.title("x reconstruct image")
        #         plt.subplot(243)
        #         plt.imshow(denorm(t_img[0]).squeeze().cpu().numpy(), cmap='gray')
        #         plt.title("original target")
        #         plt.subplot(244)
        #         plt.imshow(denorm(t_fake[0]).squeeze().cpu().numpy(), cmap='gray')
        #         plt.title("fake target")
        #         plt.show()
        #         plotted = 1
        #         plt.close(fig)

        for k in range(c_trg.size(0)):
            filename = os.path.join("Segmentation",
                                    '%.4i_%.2i.png' % (args.sepoch * args.eval_batch_size + (k + 1), epoch + 1))
            save_image(t_fake[k], ncol=1, filename=filename)
            filename = os.path.join("Ground",
                                    '%.4i_%.2i.png' % (args.sepoch * args.eval_batch_size + (k + 1), epoch + 1))
            save_image(t_img[k], ncol=1, filename=filename)


'''
Giov FID
'''

@torch.no_grad()
def calculate_FID_Giov(nets, args, step, mode):
    print('Calculating evaluation metrics...')
    domains = os.listdir(args.val_img_dir)
    domains.sort()
    num_domains = len(domains)
    #calculate_fid_for_all_tasks(args, domains, step=step, mode=mode)
    print('Number of domains: %d' % num_domains)
    lpips_dict = OrderedDict()
    loaders = {
        "t1_loader":DataLoader(syneval_dataset,batch_size=args.eval_batch_size),
        "t2_loader": DataLoader(syneval_dataset2,batch_size=args.eval_batch_size),
        "ct_loader":DataLoader(syneval_dataset3,batch_size=args.eval_batch_size)
    }
    mod = {"t1":0,"t2":1,"ct":2}

    #loaders = (syneval_dataset, syneval_dataset2,syneval_dataset3)
    #loaders = (DataLoader(syneval_dataset,batch_size=4), DataLoader(syneval_dataset2,batch_size=4),DataLoader(syneval_dataset3,batch_size=4))

    for trg_idx, trg_domain in tqdm(enumerate(domains)):
        src_domains = [x for x in domains if x != trg_domain]
        loader_ref = loaders[trg_domain+"_loader"]
        path_ref = os.path.join(args.png_dataset_path+'/eval', trg_domain)
        # loader_ref = get_eval_loader(root=path_ref,
        #                                  img_size=args.image_size,
        #                                  batch_size=args.eval_batch_size,
        #                                  imagenet_normalize=False,
        #                                  drop_last=True)
        for src_idx, src_domain in enumerate(src_domains):
            loader_src = loaders[src_domain+"_loader"]
            path_src = os.path.join(args.png_dataset_path+'/eval', src_domain)
            # loader_src = get_eval_loader(root=path_src,
            #                              img_size=args.image_size,
            #                              batch_size=args.eval_batch_size,
            #                              imagenet_normalize=False)
            task = '%s to %s' % (src_domain, trg_domain)
            path_fake = os.path.join(args.eval_dir, task)
            shutil.rmtree(path_fake, ignore_errors=True)
            os.makedirs(path_fake)
            lpips_values = []
            for i, (x_src,x_msk) in enumerate(tqdm(loader_src, total=len(loader_src))):
                N = x_src.size(0)
                x_src = x_src.to(device)
                #y_trg = torch.tensor([trg_idx] * N).to(device)
                # generate 10 outputs from the same input
                group_of_images,ground_of_images = [],[]
                for j in range(10): #num outs per domain
                    try:
                        x_ref = next(iter_ref).to(device)
                    except:
                        iter_ref = iter(loader_ref)
                        x_ref,x_ref_msk = next(iter_ref)
                        x_ref = x_ref.to(device)

                    if x_ref.size(0) < N:
                        # x_ref = x_ref[:N]
                        print(x_ref.shape)
                        break
                    #idx = random.choice([0,1,2]) #parte da zoomare?
                    #x_src_batch = x_src.unsqueeze(0)
                    idx = mod[trg_domain]
                    c = getLabel(x_src, device, idx, args.c_dim)
                    x_src = x_src[:,:1,:,:]
                    x_fake = nets.netG_use(x_src, None, c, mode='test')
                    group_of_images.append(x_fake)
                    ground_of_images.append(x_src)
                    # save generated images to calculate FID later
                    # for k in range(N):
                    #     filename = os.path.join(path_fake, '%.4i_%.2i.png' % (i*args.eval_batch_size+(k+1), j+1))
                    #     save_image(x_fake[k], ncol=1, filename=filename)

                lpips_value = calculate_FID_Giovanni_given_images(group_of_images,ground_of_images)
                if lpips_value != 0:
                    lpips_values.append(lpips_value)
                else:
                    print("MANNAGGIA")
            #print(lpips_values)
            # calculate LPIPS for each task (e.g. cat2dog, dog2cat)
            lpips_mean = np.array(lpips_values).mean()
            lpips_dict['FID_giov_%s/%s' % (mode, task)] = lpips_mean

        # delete dataloaders
        del loader_src
        if mode == 'test':
            del loader_ref
            del iter_ref

    # calculate the average LPIPS for all tasks
    lpips_mean = 0
    for _, value in lpips_dict.items():
        lpips_mean += value / len(lpips_dict)
    lpips_dict['FID_giov_%s/mean' % mode] = lpips_mean

    # report LPIPS values
    filename = os.path.join(args.eval_dir, 'FID_giov_%.5i_%s.json' % (step, mode))
    save_json(lpips_dict, filename)

    # calculate and report fid values
    return lpips_dict


# prepare the 3D model
class TargetNet(nn.Module):
    def __init__(self, base_model, n_class=1):
        super(TargetNet, self).__init__()

        self.base_model = base_model
        self.dense_1 = nn.Linear(512, 1024, bias=True)
        self.dense_2 = nn.Linear(1024, n_class, bias=True)

    def forward(self, x):
        self.base_model(x)
        self.base_out = self.base_model.out512
        self.skip_out = self.base_model.skip_out512
        # [1, 512, 16, 16, 2]
        # self.base_out = self.base_out.reshape(1,512,2,16,16)
        # This global average polling is for shape (N,C,H,W) not for (N, H, W, C)
        # where N = batch_size, C = channels, H = height, and W = Width
        self.out_glb_avg_pool = F.avg_pool3d(self.base_out,
                                             kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0], -1)
        return self.out_glb_avg_pool
        # self.linear_out = self.dense_1(self.out_glb_avg_pool)
        # final_out = self.dense_2( F.relu(self.linear_out))
        # return final_out


base_model = unet3d.UNet3D()

# Load pre-trained weights
weight_dir = 'pretrained_weights/Genesis_Chest_CT.pt'
checkpoint = torch.load(weight_dir, map_location=torch.device("cpu"))
state_dict = checkpoint['state_dict']
unParalled_state_dict = {}
for key in state_dict.keys():
    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
base_model.load_state_dict(unParalled_state_dict)
target_model = TargetNet(base_model)
target_model.to("cpu")
target_model = nn.DataParallel(target_model, device_ids=[i for i in range(torch.cuda.device_count())])
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(target_model.parameters(), 1, momentum=0.9, weight_decay=0.0, nesterov=False)

# train the model
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(50 * 0.8), gamma=0.5)

# for epoch in tqdm(range(0, 10000)):
#     scheduler.step(epoch)
#     target_model.train()
#     for batch_ndx, x in enumerate(tensors):
#         x = x.float().to("cpu")
#         print(x.shape)
#         #x = x.unsqueeze(dim=1)
#         pred = F.sigmoid(target_model(x))
#         loss = criterion(pred, y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

from scipy import linalg


def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu - mu2) ** 2) + np.trace(cov + cov2 - 2 * cc)
    return np.real(dist)


def calculate_FID_Giovanni_given_images(group_of_images, ground_of_images):
    fea_fake, fea_real = [], []

    for x_real, x_fake in zip(group_of_images, ground_of_images):
        if x_real.size(0) < 16:
            return 0
        x_real, x_fake = x_real.reshape(1, 128, 128, 16), x_fake.reshape(1, 128, 128, 16)
        x_real, x_fake = x_real.unsqueeze(1).float(), x_fake.unsqueeze(1).float()
        fea_real.append(target_model(x_real))
        fea_fake.append(target_model(x_fake))
    mu, cov = [], []
    fea_fake = torch.cat(fea_fake)
    fea_real = torch.cat(fea_real)
    # print(fea_fake.shape)
    mu.append(np.mean(fea_real.cpu().detach().numpy(), axis=0))
    mu.append(np.mean(fea_fake.cpu().detach().numpy(), axis=0))
    cov.append(np.cov(fea_real.cpu().detach().numpy(), rowvar=False))
    cov.append(np.cov(fea_fake.cpu().detach().numpy(), rowvar=False))

    fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
    return fid_value


def calculate_all_metrics(fid_png=False):
    _, fid_stargan = calculate_metrics(nets, args, args.sepoch, args.experiment_name)
    if fid_png:
        fid_dict = calculate_fid()
    else:
        fid_dict = {}
    mod = ["t1", "t2", "ct"]
    fid_giov = calculate_FID_Giov(nets, args, args.sepoch, args.experiment_name)
    dice_dict, ravd_dict, s_score_dict = {}, {}, {}
    for i in range(3):
        # ======= Directories =======
        cwd = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        ground_dir = os.path.normpath('Ground')
        seg_dir = os.path.normpath('Segmentation')
        dicom_dir = os.path.normpath(cwd + '/Data_3D/DICOM_anon')

        eval_dice_or_s_score(i, True)
        # ======= Volume Reading =======
        Vref = png_series_reader(ground_dir)
        Vseg = png_series_reader(seg_dir)
        print('Volumes imported.')
        # ======= Evaluation =======
        print('Calculating for  modality ...', mod[i])
        dice, ravd = evaluate(Vref, Vseg, dicom_dir)
        dice_dict["DICE/" + mod[i]] = dice
        ravd_dict["RAVD/" + mod[i]] = ravd

        # calculate s score
        eval_dice_or_s_score(i, False)
        # ======= Volume Reading =======
        Vref = png_series_reader(ground_dir)
        Vseg = png_series_reader(seg_dir)
        [dice, ravd] = evaluate(Vref, Vseg, dicom_dir)
        s_score_dict["S-SCORE/" + mod[i]] = dice

        # if dice_:
        #     print('DICE=%.3f RAVD=%.3f ' %(dice, ravd))
        # else:
        #     print('S-score = %.3f' %(dice))

    return fid_stargan, fid_dict, dice_dict, ravd_dict, s_score_dict, fid_giov

