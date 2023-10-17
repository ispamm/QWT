from pytorch_wavelets import DWTForward,DWTInverse
from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode
from qwt import QWTForward, QWTInverse
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics.functional import peak_signal_noise_ratio,structural_similarity_index_measure
from torch.utils.data import DataLoader
# from marina_code.get_data import DatasetForCAE_IXI, DatasetKvasir
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"

## Residual Block (RCB)
class ResidualBlock(nn.Module):
    def __init__(self, n_feat, out_channels, kernel_size, res_scale=1):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
                nn.Conv2d(in_channels=n_feat, out_channels = n_feat, kernel_size = kernel_size, padding=1 ),
                nn.ReLU(),
                nn.Conv2d(in_channels=n_feat, out_channels = out_channels, kernel_size = kernel_size, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, n_feat, out_channels, kernel_size, n_resblocks=3):
        super(ResidualGroup, self).__init__()
        modules_body = [
            ResidualBlock(n_feat = n_feat, out_channels = out_channels, kernel_size = kernel_size) \
            for _ in range(n_resblocks)]
        self.modules_body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.modules_body(x)
        res += x
        return res


class RoughFeatureExtractionModule(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, n_residual_groups=3):
        super(RoughFeatureExtractionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1)
        self.residual_groups = [ 
            ResidualGroup(n_feat=64, out_channels = out_channels, kernel_size=kernel_size)
            for _ in range(n_residual_groups)
        ]
        self.residual_groups = nn.Sequential(*self.residual_groups)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=kernel_size, padding=1)
    def forward(self,x):
        shallow_feature = self.conv1(x)
        f_rg = self.residual_groups(shallow_feature)
        f_rough = self.conv2(f_rg) + shallow_feature
        # if torch.isnan(f_rough).any():
        #     raise Exception("nan rfem")
        return f_rough

class WaveletTransformModule(nn.Module):
    def __init__(self, in_channels=64, out_channels=1, kernel_size=3, espcn=False, wavelet_type="dwt", double_last_layer=False):
        super(WaveletTransformModule, self).__init__()
        self.wavelet_type = wavelet_type
        self.double_last_layer = double_last_layer #or out_channels == 3
        self.split_size = out_channels * 1 if self.wavelet_type =='dwt' else out_channels * 4
        self.wavelet_layer = DWTForward(J=1, wave='bior1.3', mode='zero') if wavelet_type =='dwt' else QWTForward(device)
        self.feature_refinement_ll = ResidualGroup(n_feat = in_channels, out_channels = in_channels, kernel_size = kernel_size)
        self.feature_refinement_lh = ResidualGroup(n_feat = in_channels, out_channels = in_channels, kernel_size = kernel_size)
        self.feature_refinement_hl = ResidualGroup(n_feat = in_channels, out_channels = in_channels, kernel_size = kernel_size)
        self.feature_refinement_hh = ResidualGroup(n_feat = in_channels, out_channels = in_channels, kernel_size = kernel_size)
        
        if self.wavelet_type == "qwt":
            self.shrink_conv0 = nn.Conv2d(in_channels=in_channels*4, out_channels=in_channels, kernel_size=1, padding=0)
            self.shrink_conv1 = nn.Conv2d(in_channels=in_channels*4, out_channels=in_channels, kernel_size=1, padding=0)
            self.shrink_conv2 = nn.Conv2d(in_channels=in_channels*4, out_channels=in_channels, kernel_size=1, padding=0)
            self.shrink_conv3 = nn.Conv2d(in_channels=in_channels*4, out_channels=in_channels, kernel_size=1, padding=0)


        self.upsample = nn.Upsample(scale_factor=2)
        self.espcn = espcn
        if espcn:
            self.upsample = self.load_espcn()

        self.feature_upsample_lh = ResidualGroup(n_feat = in_channels*2, out_channels = in_channels*2, kernel_size = kernel_size)
        self.feature_upsample_hl = ResidualGroup(n_feat = in_channels*2, out_channels = in_channels*2, kernel_size = kernel_size)
        self.feature_upsample_hh = ResidualGroup(n_feat = in_channels*2, out_channels = in_channels*2, kernel_size = kernel_size)


        self.reconstruction_ll = ResidualGroup(n_feat = in_channels, out_channels = in_channels, kernel_size = kernel_size)
        self.reconstruction_lh = ResidualGroup(n_feat = in_channels,out_channels = in_channels, kernel_size = kernel_size)
        self.reconstruction_hl = ResidualGroup(n_feat = in_channels,out_channels = in_channels,kernel_size = kernel_size)
        self.reconstruction_hh = ResidualGroup(n_feat = in_channels,out_channels = in_channels,kernel_size = kernel_size)

        out_chan = out_channels if self.wavelet_type =='dwt' else 4 *out_channels
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=out_chan, kernel_size=1, padding=0)
        self.conv1_0 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=kernel_size, padding=0)
        self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_chan, kernel_size=1, padding=1)
        self.conv2_0 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=kernel_size, padding=0)
        self.conv2_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_chan, kernel_size=1, padding=1)
        self.conv3_0 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=kernel_size, padding=0)
        self.conv3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_chan, kernel_size=1, padding=1)

        out_chan_weigh = out_channels * 4 if self.wavelet_type =='dwt' else out_channels * 16 
        self.last_weights = nn.Conv2d(in_channels=out_chan_weigh, out_channels=out_chan_weigh, kernel_size=1)
        if self.double_last_layer:
            self.last_weights = nn.Conv2d(in_channels=out_chan_weigh, out_channels=4, kernel_size=1)
            self.qwt_last_weights = nn.Conv2d(in_channels=4, out_channels=out_chan_weigh, kernel_size=1) #if self.wavelet_type =='qwt' else None

            # self.qwt_last_weights = nn.Conv2d(in_channels=out_chan_weigh, out_channels=out_chan_weigh, kernel_size=1) if self.wavelet_type =='qwt' else None
        
        self.inverse_wavelet_layer = DWTInverse(wave='bior1.3',mode='zero') if self.wavelet_type =='dwt' else QWTInverse(device)
        
    def edge_extraction_module(self,x, LH,HL,HH):
        F_efm = torch.sqrt(LH*LH+HL*HL+HH*HH)
        return torch.cat((x,F_efm),dim=1)
    
    def apply_wavelet_features(self,features):
        if self.wavelet_type=="dwt":
            image_size = features.shape[2]
            resize_function = Resize((image_size  - 4, image_size  - 4), interpolation=InterpolationMode.BICUBIC)  # Resize((image_size-3, image_size-3)) 
            features = resize_function(features)
        LL, Yh = self.wavelet_layer(features)
        LH, HL, HH = Yh[0][:,:,0], Yh[0][:,:,1], Yh[0][:,:,2]
        LL = (LL-LL.min())/(LL.max()-LL.min())
        LH,HL,HH = (LH-LH.min())/(LH.max()-LH.min()), (HL-HL.min())/(HL.max()-HL.min()), (HH-HH.min())/(HH.max()-HH.min())
        return LL,LH,HL,HH

    def forward(self,features):
        # if torch.isnan(features).any():
        #     raise Exception("nan first wav")
        
        LL,LH,HL,HH = self.apply_wavelet_features(features)

        if self.wavelet_type != 'dwt':
            LL = self.shrink_conv0(LL)
            LH = self.shrink_conv1(LH) 
            HL = self.shrink_conv2(HL) 
            HH = self.shrink_conv3(HH) 

        LL = self.feature_refinement_ll(LL)
        LH_res = self.feature_refinement_lh(LH)
        HL_res = self.feature_refinement_hl(HL)
        HH_res = self.feature_refinement_hh(HH)

        LH_edge = self.edge_extraction_module(LH_res,LH,HL,HH)
        HL_edge = self.edge_extraction_module(HL_res,LH,HL,HH)
        HH_edge = self.edge_extraction_module(HH_res,LH,HL,HH)

        if self.espcn:
            LH_pre_recon = self.upsample(self.upsample.preprocess(self.feature_upsample_lh(LH_edge))).clamp(0.0, 1.0)
            HL_pre_recon = self.upsample(self.upsample.preprocess(self.feature_upsample_hl(HL_edge))).clamp(0.0, 1.0)
            HH_pre_recon = self.upsample(self.upsample.preprocess(self.feature_upsample_hh(HH_edge))).clamp(0.0, 1.0)
            LL_up = self.upsample(self.upsample.preprocess(LL)).clamp(0.0, 1.0)
            features_up = nn.Upsample(scale_factor=1.5)(features)
        else:
            LH_pre_recon = self.upsample(self.feature_upsample_lh(LH_edge)).clamp(0.0, 1.0)
            HL_pre_recon = self.upsample(self.feature_upsample_hl(HL_edge)).clamp(0.0, 1.0)
            HH_pre_recon = self.upsample(self.feature_upsample_hh(HH_edge)).clamp(0.0, 1.0)

        LL_up = self.upsample(LL).clamp(0.0, 1.0)
        features_up = features #self.upsample(features)
        LL_pre_recon = features_up - LL_up

        component_0 = features_up
        component_1 = torch.cat([LL_pre_recon,LH_pre_recon],dim=1)
        component_2 = torch.cat([LL_pre_recon,HL_pre_recon],dim=1)
        component_3 = torch.cat([LL_pre_recon,HH_pre_recon],dim=1)
        LL = self.conv0(self.reconstruction_ll(component_0))
        LH = self.conv1_1(self.reconstruction_lh(self.conv1_0(component_1)))
        HL = self.conv2_1(self.reconstruction_hl(self.conv2_0(component_2)))
        HH = self.conv3_1(self.reconstruction_hh(self.conv3_0(component_3)))


        weights = self.last_weights(torch.cat((LL,LH,HL,HH),dim=1))
        if self.double_last_layer:
            weights = self.qwt_last_weights(weights)
        (LL,LH,HL,HH) = torch.split(weights,split_size_or_sections=self.split_size,dim=1)

        Yh = [torch.stack((LH,HL,HH),dim=2)]
        # if torch.isnan(self.inverse_wavelet_layer((LL,Yh))).any():
        #     raise Exception("nan last wav")
        res = self.inverse_wavelet_layer((LL,Yh))
        return res

    def load_espcn(self):
        model_espcn = ESPCN(scale_factor=3).to(device)
        weights_file_path = "results/espcn_x3.pth"
        state_dict = model_espcn.state_dict()
        for n, p in torch.load(weights_file_path, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)
        model_espcn.eval()
        return model_espcn

class IXIDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './IXI_preprocess_dataset', batch_size: int = 16, seed=888):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed
    def setup(self,stage=None):
        self.dataset_train = DatasetForCAE_IXI(self.data_dir, transform=None, train=True, seed=self.seed)
        self.dataset_valid = DatasetForCAE_IXI(self.data_dir, transform=None, train=False, seed=self.seed)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size, shuffle=False, num_workers=4)

class KvasirDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/var/datasets/Kvasir", batch_size: int = 16, seed=888):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed
    def setup(self,stage=None):
        self.dataset_train = DatasetKvasir(self.data_dir, subset='train', seed=self.seed)
        self.dataset_valid = DatasetKvasir(self.data_dir, subset='eval', seed=self.seed)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size, shuffle=False, num_workers=4)

#from celeba import CelebADataset

class CelebADataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = '/home/luigi/Documents/swagan/datasets/celeba_1024', 
                        batch_size: int = 16, seed=888, image_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed
        self.image_size = image_size
    def setup(self,stage=None):
        full_dataset = CelebADataset(self.data_dir, self.image_size)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        self.dataset_train, self.dataset_valid = torch.utils.data.random_split(full_dataset, [train_size, test_size],
                                                                                generator=torch.Generator().manual_seed(self.seed))


    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size, shuffle=False, num_workers=4)





class ImageFusionNetworkPL(pl.LightningModule):

    def __init__(self, 
                learning_rate=1e-4,
                in_channels=1,
                out_channels=1,
                wavelet_type="qwt",
                double_last_layer=False
                ):
        super().__init__()
        self.save_hyperparameters()
        self.l1 = nn.L1Loss()
        self.RFEM = RoughFeatureExtractionModule(in_channels=in_channels, out_channels=64, kernel_size=3, n_residual_groups=3)
        self.WTM = WaveletTransformModule(in_channels=64,out_channels=out_channels, kernel_size=3, espcn=False, wavelet_type=wavelet_type,double_last_layer=double_last_layer)
        self.res_128 = Resize((128,128),interpolation=InterpolationMode.BICUBIC)
        self.res_252 = Resize((252,252),interpolation=InterpolationMode.BICUBIC)
        self.lr = learning_rate
        self.wavelet_type = wavelet_type
        self.wavelet_layer = DWTForward(J=1, wave='bior1.3', mode='zero') if self.wavelet_type =='dwt' else QWTForward(self.device)

    def forward(self, x):
        features = self.RFEM(x)
        return self.WTM(features) 

    def training_step(self, batch, batch_idx):
        data_128 = self.res_128(batch)
        data_252 = self.res_252(batch)
        
        pred = self(data_128)
        orig = data_252 if self.wavelet_type =='dwt' else batch
        loss,cont_loss,wav_loss = self.loss_function(orig, pred)
        values = {"loss_tot": loss, "cont_loss": cont_loss, "wav_loss": wav_loss}  # add more items if needed
        self.log_dict(values,prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # lr_scheduler = MultiStepLR(optimizer, milestones=[50,100,150,250,300,350], gamma=0.5)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=7, verbose=True)
        return [optimizer], {"scheduler": lr_scheduler, "monitor": "psnr"}

    def validation_step(self, batch, batch_idx):
        data_128 = self.res_128(batch)
        data_252 = self.res_252(batch)
        orig = data_252 if self.wavelet_type =='dwt' else batch
        pred = self(data_128)
        # print(pred.shape, orig.shape)
        ssim = (structural_similarity_index_measure(pred, orig))
        psnr = (peak_signal_noise_ratio(pred, orig))
        loss,cont_loss,wav_loss = self.loss_function(orig, pred)
        
        values = {"psnr": psnr, "ssim": ssim, "val_loss_tot": loss}  # add more items if needed
        self.log_dict(values,prog_bar=True)
        
        wandb.log({"original": wandb.Image(orig[0])}, commit=False)
        wandb.log({"pred": wandb.Image(pred[0])}, commit=False)


    def loss_function(self,x,pred,alpha=1):
        content_loss = self.l1(pred, x)

        wavelet_x = self.resize_and_extract_wav(x)
        wavelet_pred = self.resize_and_extract_wav(pred)
        wavelet_loss = self.l1(wavelet_pred, wavelet_x)

        loss = content_loss  + alpha*wavelet_loss
        return loss,content_loss, wavelet_loss

    def resize_and_extract_wav(self,img):
        if self.wavelet_type =='dwt':
            image_size = img.shape[2]
            resize_function = Resize((image_size  - 4, image_size  - 4),interpolation=InterpolationMode.BICUBIC)
            img = resize_function(img)
        
        LL, Yh = self.wavelet_layer(img)
        LH, HL, HH = Yh[0][:,:,0], Yh[0][:,:,1], Yh[0][:,:,2]
        LL = (LL-LL.min())/(LL.max()-LL.min())
        LH,HL,HH = (LH-LH.min())/(LH.max()-LH.min()), (HL-HL.min())/(HL.max()-HL.min()), (HH-HH.min())/(HH.max()-HH.min())
        return torch.cat((LL, LH, HL, HH),dim=1)



