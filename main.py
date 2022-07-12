import os

import torch
import random
import numpy as np

from config import args
from dataset import ChaosDataset_Syn_new
from metrics import compute_miou, create_images_for_dice_or_s_score, evaluate, evaluation, calculate_ignite_inception_score, png_series_reader
from train import train
from sample import sample
from utils import build_model, load_nets
from tqdm import tqdm
def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


# device = torch.device("cuda:1,3" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.


if __name__ == '__main__':
    set_deterministic(args.seed)
    print(args)
    if args.mode == "train":
        train(args)
    if args.mode == "eval":
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        evaluation()

    if args.mode == "sample":
        sample()