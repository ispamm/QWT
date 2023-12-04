import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy          as np
from numpy import savez, asarray
from sys              import exit
# from scipy.misc       import imread
from imageio import imread
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import pywt
import pywt.data
from scipy.fftpack import hilbert as ht
from six.moves import xrange
from pathlib import Path
import torch
import torchvision
from PIL import Image

# import sys
# sys.path.append('/home/luigi/Documents/tests/')

# wavelet_type="qwt"
# device = "cuda" if torch.cuda.is_available() else 'cpu'
# from image_fusion import ImageFusionNetwork
# model_IF = ImageFusionNetwork(in_channels=1, out_channels=1,wavelet_type=wavelet_type).to(device)
# str_path = "/home/luigi/Documents/tests/results/checkpoints"+wavelet_type+"_IFN_" + str(400) + ".pt"
# model_IF.load_state_dict(torch.load(str_path))
# model_IF.eval()

# from pl_image_fusion import ImageFusionNetworkPL
# model_IF = ImageFusionNetworkPL.load_from_checkpoint("/home/luigi/Documents/tests/results/checkpoints/IFN-IXI-qwt-ixi-qwt-bicubic-epoch=399-psnr=29.86.ckpt").to(device)
# model_IF.eval()

# def image_fusion_preprocess(img):
#     def get_activation(name,activation_json):
#         def hook(model, input, output):
#             activation_json[name] = output.detach()
#         return hook

#     activation_json = {}
#     # if wavelet_type =="dwt":
#     return_string = "WTM.last_weights"
#     model_IF.WTM.last_weights.register_forward_hook(get_activation(return_string,activation_json))
#     # else:
#     #     return_string = 'WTM.qwt_last_weights'
#     #     model_IF.WTM.qwt_last_weights.register_forward_hook(get_activation(return_string,activation_json))

#     output = model_IF(img)

#     return activation_json[return_string]


# def wavelet_transformation(img):
    
#     # Wavelet transform of image
#     #titles = ['Approximation', ' Horizontal detail','Vertical detail', 'Diagonal detail']
    
#     coeffs2 = pywt.dwt2(img, 'bior1.3') #Biorthogonal wavelet
#     LL, (LH, HL, HH) = coeffs2
#     # fig = plt.figure(figsize=(12, 3))
#     # for i, a in enumerate([LL, LH, HL, HH]):
#     #         ax = fig.add_subplot(1, 4, i + 1)
#     #         ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#     #         ax.set_title(titles[i], fontsize=10)
#     #         ax.set_xticks([])
#     #         ax.set_yticks([])
    
#     # fig.tight_layout()
#     # plt.show()
    
#     return LL, LH, HL, HH


class DatasetKvasir(Dataset):
    """Brain MRI dataset for FLAIR abnormality segmentation"""

    def __init__(
        self,
        images_dir,
        subset = "train",
        seed = 13,
    ):
        
        print("reading {} images...".format(subset))
        
        if images_dir == "/var/datasets/Kvasir":
            
            src_images = images_dir +'/*/*_image.jpg'
            #src_masks = images_dir +'/*/*_mask.jpg'
        
            path_images = glob.glob(src_images) 
            #path_masks = glob.glob(src_masks) 
        
            #path_images.sort(key=myFunc)
            #path_masks.sort(key=myFunc)
        
            self.train_X, self.validation_X = train_test_split(path_images,test_size=0.25,train_size=0.75, random_state=seed, shuffle=False)       
            #self.train_y, self.validation_y = train_test_split(path_masks,test_size=0.25,train_size=0.75, random_state=seed, shuffle=False)

            if subset == "train":
                self.input = self.train_X
                #self.target = self.train_y
            
            else :
                self.input = self.validation_X
                #self.target = self.validation_y


    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        
        image = self.input[idx]
        #mask = self.target[idx]
        
        ########## IMAGE ###############
        image = imread(image)
        image = cv2.resize(image, (256, 256)) #array of uint8
        image = image/255.0

        ########## MASK ###############                             
        # mask = imread(mask, as_gray=True)
        # mask = cv2.resize(mask, (256, 256))
        # mask = (mask>0.5).astype('float32')

        image = np.array(image)
        #mask = np.array(mask)
        
        #mask = np.expand_dims(mask, axis=0)
        image = image.transpose(2,0,1)
        
        #print("image 1-->", np.shape(image), type(image))
        #print("mask 1-->" ,np.shape(mask), type(mask))
        
        '''
        if self.transform is not None:
            print("\n\n\n ENTRO ENTRO ENTRO \n\n\n")
            image, mask = self.transform((image, mask))
        '''
        
        #image = image.transpose(2, 0, 1)
        #mask = mask.transpose(2, 0, 1)
        
        #print("image shape-->", np.shape(image), type(image))
        #print("mask shape--->", np.shape(mask), type(image))
        
        image_tensor = torch.from_numpy(image.astype(np.float32))
        #mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # return tensors
        return image_tensor#, mask_tensor


class DatasetForCAE_IXI(torch.utils.data.Dataset):

    'Characterizes a dataset for PyTorch'

    def __init__(
        self,
        path,
        train,
        transform,
        seed=42
    ):
    
        # path = "./IXI_preprocess_dataset"
        
        path1 = '../../data/IXI_preprocess_dataset/IXI-T1/T1.npz'#path + "/IXI-T1/T1.npz" 
        path2 = '../../data/IXI_preprocess_dataset/IXI-T2/T2.npz'#path + "/IXI-T2/T2.npz"
        
        self.images1 = np.load(path1,allow_pickle=True)['arr_0'] # array of uint8 (581, 256,256) valori [0,255]
        self.images2 = np.load(path2,allow_pickle=True)['arr_0'] # array of uint8 (578, 256,256) valori [0,255]
        
        self.train, self.validation = train_test_split(self.images1,test_size=0.25,train_size=0.75,random_state=seed) #train (435,256,256)  validation (146,256,256)        
        self.train2, self.validation2 = train_test_split(self.images2,test_size=0.25,train_size=0.75, random_state=seed) #train2 (433,256,256)
        
        
        
        if train:

            # read images
            print("\nreading train images...")
            
            self.train = self.train[:-2,:,:]
            
            self.input = np.concatenate((self.train,self.train2), axis=0) #(866, 256, 256)
            self.input = self.input[np.random.permutation(self.input.shape[0]),:]


        else:
                
            # read images
            print("reading validation images...\n")
            
            self.validation = self.validation[:-1,:,:]
            
            self.input = np.concatenate((self.validation,self.validation2), axis=0) #(290, 256, 256)
            self.input = self.input[np.random.permutation(self.input.shape[0]),:]

            
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.input)
              
    def __getitem__(self, index):
        'Generates one sample of data'
        
        # Select sample
        x = self.input[index] #(256,256)
        
        x = x/255 #[0,1]
        # x = np.repeat(x[:, :, np.newaxis], 3, axis=2)
        # x = np.reshape(x, (x.shape[2], x.shape[0], x.shape[1])) # array of uint8 (3,256,256) valori [0,1] 
        # x = np.transpose(x, (2,0,1)) # array of uint8 (3,256,256) valori [0,1]
        
        x = torch.from_numpy(x.astype(np.float32))
        return x.unsqueeze(0)

   
   

class CelebADataset(torch.utils.data.Dataset):

    def __init__(self, path: str, image_size: int):

        super().__init__()

        # Get the paths of all `jpg` files
        self.paths = [p for p in Path(path).glob(f'**/*.jpg')]

        # Transformation
        self.transform = torchvision.transforms.Compose([
            # Resize the image
            torchvision.transforms.Resize(image_size),
            # Convert to PyTorch tensor
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

