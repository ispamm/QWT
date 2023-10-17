import os
import wave

from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import shutil
import SimpleITK as sitk
import cv2
import pydicom
import torch
import random
from torchvision import transforms
import pywt
import pywt.data
from itertools import chain
from pathlib import Path
from configs.config_tmp import args
from scipy.fftpack import hilbert as ht
from six.moves import xrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from imageio import imread

class ChaosDataset_Syn_new(Dataset):
    def __init__(self, path="../datasets/chaos2019", split='train', modals=('t1', 't2', 'ct'), transforms=None,
                 image_size=256):
        super(ChaosDataset_Syn_new, self).__init__()
        for modal in modals:
            assert modal in {'t1', 't2', 'ct'}
        fold = split + "/"
        path1 = os.path.join(path, fold + modals[0])
        path2 = os.path.join(path, fold + modals[1])
        path3 = os.path.join(path, fold + modals[2])
        list_path = sorted([os.path.join(path1, x) for x in os.listdir(path1)]) + sorted(
            [os.path.join(path2, x) for x in os.listdir(path2)])
        raw_path = []
        label_path = []
        print(list_path)

        for x in list_path:
            if "t1" in x:
                # x += "/T1DUAL"
                c = np.array(0)
            elif "t2" in x:
                # x += "/T2SPIR"
                c = np.array(1)
            for y in os.listdir(x):
                if "Ground" in y:
                    tmp = os.path.join(x, y)
                    raw_path.append([tmp.replace("Ground", "DICOM_anon"), c])
                    break
        #########
        self.raw_dataset = []
        self.label_dataset = []
        #######
        self.transfroms = transforms
        self.image_size = image_size

        for i, c in tqdm(raw_path):
            if c == 0:
                i += "/InPhase"
                for y in os.listdir(i):
                    tmp = os.path.join(i, y)
                    img = sitk.ReadImage(tmp)
                    img = sitk.GetArrayFromImage(img)[0]
                    self.raw_dataset.append([raw_preprocess(img, True), c])
                    a = tmp.replace("DICOM_anon/InPhase", "Ground")
                    img = sitk.ReadImage(a.replace(".dcm", ".png"))  # ground is a png file
                    img = sitk.GetArrayFromImage(img)
                    self.label_dataset.append(label_preprocess(img))
            elif c == 1:
                for y in os.listdir(i):
                    tmp = os.path.join(i, y)
                    img = sitk.ReadImage(tmp)
                    img = sitk.GetArrayFromImage(img)[0]
                    self.raw_dataset.append([raw_preprocess(img, True), c])
                    a = tmp.replace("DICOM_anon", "Ground")
                    img = sitk.ReadImage(a.replace(".dcm", ".png"))
                    img = sitk.GetArrayFromImage(img)
                    self.label_dataset.append(label_preprocess(img))
        list_path = sorted([os.path.join(path3, x) for x in os.listdir(path3)])
        raw_path = []
        assert len(raw_path) == 0
        for x in list_path:
            c = np.array(2)
            for y in os.listdir(x):
                if "Ground" in y:
                    tmp = os.path.join(x, y)
                    label_path.append(tmp)
                    raw_path.append([tmp.replace("Ground", "DICOM_anon"), c])
                    break
        for i, c in tqdm(raw_path):
            for y in sorted(os.listdir(i)):
                tmp = os.path.join(i, y)
                dcm = pydicom.dcmread(tmp)
                wc = dcm.WindowCenter[0]
                ww = dcm.WindowWidth[0]
                slope = dcm.RescaleSlope
                intersept = dcm.RescaleIntercept
                low = wc - ww // 2
                high = wc + ww // 2
                img = dcm.pixel_array * slope + intersept
                img[img < low] = low
                img[img > high] = high
                img = (img - low) / (high - low)
                shape = img.copy()
                shape[shape != 0] = 1
                self.raw_dataset.append([[img, shape], c])

        for i in tqdm(label_path):
            for y in sorted(os.listdir(i)):
                img = sitk.ReadImage(os.path.join(i, y))
                img = sitk.GetArrayFromImage(img)
                data = img.astype(dtype=int)
                new_seg = np.zeros(data.shape, data.dtype)
                new_seg[data != 0] = 1
                self.label_dataset.append(new_seg)
        self.split = split
        assert len(self.raw_dataset) == len(self.label_dataset)
        print("chaos train data load success!")
        print("modal:{},fold:{}, total size:{}".format(modals, fold, len(self.raw_dataset)))
    def __len__(self):
        return len(self.raw_dataset)
    def __getitem__(self, item):
        img, shape_mask, class_label, seg_mask = self.raw_dataset[item][0][0], self.raw_dataset[item][0][1], \
                                                 self.raw_dataset[item][1], self.label_dataset[item]
        if img.shape[0] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return img
        
        
def label_preprocess(data):
    data = data.astype(dtype=int)
    new_seg = np.zeros(data.shape, data.dtype)
    new_seg[(data > 55) & (data <= 70)] = 1
    return new_seg


def raw_preprocess(data, get_s=False):
    """
    :param data: [155,224,224]
    :return:
    """
    data = data.astype(dtype=float)
    data[data < 50] = 0
    out = data.copy()
    out = (out - out.min()) / (out.max() - out.min())

    if get_s:
        share_mask = out.copy()
        share_mask[share_mask != 0] = 1
        return out, share_mask
    return out
