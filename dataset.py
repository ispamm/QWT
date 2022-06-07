"""
mask[256，256]
Liver: 63 (55<<<70)
"""
import os

from PIL import Image
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

from config import args
from utils import listdir


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

    def __getitem__(self, item):
        img, shape_mask, class_label, seg_mask = self.raw_dataset[item][0][0], self.raw_dataset[item][0][1], \
                                                 self.raw_dataset[item][1], self.label_dataset[item]
        if img.shape[0] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            seg_mask = cv2.resize(seg_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            shape_mask = cv2.resize(shape_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        t_img = img * seg_mask
        if self.split == 'train':
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                seg_mask = cv2.flip(seg_mask, 1)
                shape_mask = cv2.flip(shape_mask, 1)
                t_img = cv2.flip(t_img, 1)
        #  scale to [-1,1]
        img = (img - 0.5) / 0.5
        t_img = (t_img - 0.5) / 0.5

        return torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(dim=0), \
               torch.from_numpy(t_img).type(torch.FloatTensor).unsqueeze(dim=0), \
               torch.from_numpy(shape_mask).type(torch.LongTensor).unsqueeze(dim=0), \
               torch.from_numpy(seg_mask).type(torch.LongTensor).unsqueeze(dim=0), \
               torch.from_numpy(class_label).type(torch.FloatTensor)

    def __len__(self):
        return len(self.raw_dataset)


class ChaosDataset_Syn_Test(Dataset):

    def __init__(self, path="../datasets/chaos2019", split='test', modal='t1', gan=False, transforms=None,
                 image_size=256):
        super(ChaosDataset_Syn_Test, self).__init__()
        # assert modal in {'t1', 't2','ct'}
        fold = split + "/" + modal
        path = os.path.join(path, fold)

        list_path = sorted([os.path.join(path, x) for x in os.listdir(path)])
        raw_path = []
        label_path = []
        if gan is True:
            list_path = list_path[0:1]

        for x in list_path:
            # if modal == "t1":
            #     x += "/T1DUAL"
            # elif modal == "t2":
            #     x += "/T2SPIR"
            # print(x) #/content/drive/MyDrive/Thesis/Datasets/chaos2019/test/t1/11
            for y in os.listdir(x):
                # print(y) #outphase #inphase
                if "Ground" in y:
                    tmp = os.path.join(x, y)
                    if "ct" in x:
                        label_path.append(tmp)
                    raw_path.append(tmp.replace("Ground", "DICOM_anon"))

        self.transfroms = transforms
        self.raw_dataset = []
        self.label_dataset = []
        self.index = []
        self.image_size = image_size
        if modal == "t1":
            for i in raw_path:
                i += "/InPhase"
                n = 0
                for y in os.listdir(i):
                    tmp = os.path.join(i, y)
                    img = sitk.ReadImage(tmp)
                    img = sitk.GetArrayFromImage(img)[0]
                    self.raw_dataset.append(raw_preprocess(img))
                    a = tmp.replace("DICOM_anon/InPhase", "Ground")
                    img = sitk.ReadImage(a.replace(".dcm", ".png"))
                    img = sitk.GetArrayFromImage(img)
                    self.label_dataset.append(label_preprocess(img))
                    n += 1
                self.index.append(n)
        elif modal == 't2':
            for i in raw_path:
                n = 0
                for y in os.listdir(i):
                    tmp = os.path.join(i, y)
                    img = sitk.ReadImage(tmp)
                    img = sitk.GetArrayFromImage(img)[0]
                    self.raw_dataset.append(raw_preprocess(img))
                    a = tmp.replace("DICOM_anon", "Ground")
                    img = sitk.ReadImage(a.replace(".dcm", ".png"))
                    img = sitk.GetArrayFromImage(img)
                    self.label_dataset.append(label_preprocess(img))
                    n += 1
                self.index.append(n)
        else:
            for i in raw_path:
                n = 0
                # i += "/DICOM_anon"
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
                    self.raw_dataset.append(img)
                    n += 1
                self.index.append(n)
            for i in label_path:
                for y in sorted(os.listdir(i)):
                    img = sitk.ReadImage(os.path.join(i, y))
                    img = sitk.GetArrayFromImage(img)
                    data = img.astype(dtype=int)
                    new_seg = np.zeros(data.shape, data.dtype)
                    new_seg[data != 0] = 1
                    self.label_dataset.append(new_seg)
        self.split = split
        assert len(self.raw_dataset) == len(self.label_dataset)
        print("chaos test data load success!")
        print("modal:{},fold:{}, total size:{}".format(modal, fold, len(self.raw_dataset)))

    def __getitem__(self, item):
        img, mask = self.raw_dataset[item], self.label_dataset[item]
        if img.shape[0] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        img = (img - 0.5) / 0.5
        return torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(dim=0), torch.from_numpy(mask).type(
            torch.LongTensor)

    def __len__(self):
        return len(self.raw_dataset)

    def _getIndex(self):
        return self.index


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


class MyDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = listdir(image_paths)
        # height, width = 299, 299
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # #self.transform = transform
        self.transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        if self.transform is not None:
            x = self.transform(x)
        # x = cv2.resize(x, (128, 128), interpolation=cv2.INTER_LINEAR)

        #  scale to [-1,1]
        x = (x - 0.5) / 0.5
        return x

    def __len__(self):
        return len(self.image_paths)


class DefaultDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = DefaultDataset(root, transform=transform)
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=True,
                      drop_last=drop_last,
                      collate_fn=None
                      )

def make_dataset_consistent(mr_dir="/content/drive/MyDrive/Thesis/Datasets/chaos2019/train/MR/",
                            t1_t2_dir="/content/drive/MyDrive/Thesis/Datasets/chaos2019/train/"):
    for files in os.listdir(mr_dir):
        for name in ['T1', 'T2']:
            folder_name = "T2SPIR" if name == "T2" else "T1DUAL"
            source_dir = mr_dir + files + "/" + folder_name + "/DICOM_anon/"
            target_dir = t1_t2_dir + name + "/" + files
            shutil.move(source_dir, target_dir)
            # source_dir = mr_dir+files+"/"+folder_name+"/Ground/"
            # shutil.move(source_dir,target_dir)
# for epoch, (x_real, t_img, shape_mask, mask, label_org) in tqdm(enumerate(syn_loader),total=len(syn_loader)):

#     for k in range(x_real.size(0)):
#         if label_org[k] == torch.tensor(0):
#             filename = os.path.join(args.dataset_path+"png5050/train/t1", '%.4i_%.2i.png' % (epoch*args.batch_size+(k+1), epoch+1))
#         elif label_org[k] == torch.tensor(1):
#             filename = os.path.join(args.dataset_path+"png5050/train/t2", '%.4i_%.2i.png' % (epoch*args.batch_size+(k+1), epoch+1))
#         elif label_org[k] == torch.tensor(2):
#             filename = os.path.join(args.dataset_path+"png5050/train/ct", '%.4i_%.2i.png' % (epoch*args.batch_size+(k+1), epoch+1))

#         save_image(x_real[k], ncol=1, filename=filename)
# TESTSET
# make_dataset_consistent("/content/drive/MyDrive/Thesis/Datasets/chaos2019/test/MR/","/content/drive/MyDrive/Thesis/Datasets/chaos2019/test/")
