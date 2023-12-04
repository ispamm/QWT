from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from data import DatasetForCAE_IXI
from quave import CelebADataModule, IXIDataModule, ImageFusionNetworkPL, KvasirDataModule
from qwt import QWTInverse

if __name__ == '__main__':

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument('--experiment_name', type=str, default="new_moe_",help='')
    parser.add_argument('--wavelet_type', type=str, default="dwt",help='')
    parser.add_argument('--dataset', type=str, default="ixi",help='')
    parser.add_argument('--seed', type=int, default=888, help='seed')
    parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
    args = parser.parse_args()

    pl.seed_everything(args.seed) #set_deterministic(seed=args.seed)
    model = ImageFusionNetworkPL(in_channels=1 if args.dataset=="ixi" else 3, 
                                 out_channels=1 if args.dataset=="ixi" else 3,
                                 wavelet_type=args.wavelet_type, 
                                 double_last_layer=args.wavelet_type=="qwt", 
                                 learning_rate=args.lr
                                )
    if args.dataset =="ixi":
        ixi_data = IXIDataModule()
    elif args.dataset =="kvasir":
        ixi_data = KvasirDataModule()
    else:
        #ixi_data = CelebADataModule(data_dir='/home/luigi/Documents/swagan/datasets/FFHQ_256')
        ixi_data = CelebADataModule(data_dir='../../data/celeba_1024')

    
    wandb_logger = WandbLogger(project = "im_fusion", name=args.experiment_name)
    # early_stop_callback = EarlyStopping(monitor="psnr", min_delta=0.00, patience=3, verbose=False, mode="max",)
    checkpoint_callback = ModelCheckpoint(
        dirpath="results/checkpoints",
        filename="IFN-"+args.dataset+"-"+args.wavelet_type+'-'+args.experiment_name+"-{epoch:02d}-{psnr:.2f}",
        #every_n_epochs=20,
        every_n_train_steps=10000
    )

    torch.use_deterministic_algorithms(True, warn_only=args.wavelet_type=="dwt")

    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback],#early_stopping_callback=[early_stop_callback],
                                            logger=wandb_logger, 
                                            accelerator="gpu", 
                                            devices=1,
                                            log_every_n_steps = 10,
                                            )



    trainer.fit(model, ixi_data)





class WaveletWeights(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, wav_type, device):
        assert isinstance(wav_type, str) and wav_type in ["qwt","dwt"]
        assert isinstance(device, str) and device in ["cuda","cpu"]
        from quave import ImageFusionNetworkPL

        self.wav_type = wav_type
        self.device = device
        str_path = "pretrained_weights/IFN-IXI-epoch=399-psnr=28.73.ckpt" if self.wav_type =="qwt" \
                 else "pretrained_weights/IFN-IXI-DWT-epoch=399-psnr=24.80.ckpt"
        self.model_IF = ImageFusionNetworkPL.load_from_checkpoint(str_path).to(self.device).eval()

    def get_activation(self,name,activation_json):
        def hook(model, input, output):
            activation_json[name] = output.detach()
        return hook

    def __call__(self, sample):
        activation_json = {}
        return_string = "WTM.last_weights"
        self.model_IF.WTM.last_weights.register_forward_hook(self.get_activation(return_string,activation_json))

        _ = self.model_IF(sample)

        return activation_json[return_string]

import matplotlib.pyplot as plt
import numpy as np
import torchvision as tv
from torchvision.utils import make_grid
from torchvision.utils import save_image

# def show(tensor_image,name):
#     plt.imshow(  tensor_image.cpu().permute(1, 2, 0), cmap="gray" )
#     plt.savefig(name)

ww = WaveletWeights("dwt","cuda")
dataset = DatasetForCAE_IXI('./IXI_preprocess_dataset', transform=None, train=False, seed=888)
sample = next(iter(dataset))
# torch_sample = tv.transforms.ToTensor()(sample)[:1,]
torch_sample = tv.transforms.Resize((256,256))(sample).unsqueeze(0).cuda()

transformed_sample = ww(torch_sample)
print(transformed_sample.shape)
ww = WaveletWeights("qwt","cuda")
transformed_sample = ww(torch_sample)
print(transformed_sample.shape)

# show(torch_sample.squeeze(0),"torchsample.png")
# show(transformed_sample.squeeze(0),"wavsample.png")


# grid = make_grid(transformed_sample)
# save_image(grid, f'gridaa.png')

(LL,LH,HL,HH) = torch.split(transformed_sample,split_size_or_sections=1,dim=1)
wavelet_list = torch.cat([torch_sample, LL,LH,HL,HH],dim=0)
grid = make_grid(wavelet_list)
save_image(grid, f'grid.png')
device = 'cpu'
split_size = 12
weights = torch.randn(1,48,128,128)
inverse_wavelet_layer = QWTInverse(device)
(LL,LH,HL,HH) = torch.split(weights,split_size_or_sections=split_size,dim=1)

Yh = [torch.stack((LH,HL,HH),dim=2)]
# if torch.isnan(self.inverse_wavelet_layer((LL,Yh))).any():
#     raise Exception("nan last wav")
res = inverse_wavelet_layer((LL,Yh))
print(res.shape)