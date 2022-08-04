import os
import munch

import torch
import random
import numpy as np

# from dataset import ChaosDataset_Syn_new
# # from metrics import compute_miou, create_images_for_dice_or_s_score, evaluate, evaluation, calculate_ignite_inception_score, png_series_reader
# from utils import build_model, load_nets
from tqdm import tqdm
import importlib.util
import shutil
from sample import sample

def paper(exp_names):
    
    for _ in range(3):
        seed = random.randint(1,888)
        for file_name in tqdm(exp_names, total=len(exp_names)):
            file_path_original = os.path.join("configs",file_name)
            module_name = "configs."+file_name[:-3]
            shutil.copyfile(file_path_original, "configs/config_tmp.py")

            import sys
            import fileinput

            # This for loop scans and searches each line in the file
            # By using the input() method of fileinput module
            for line in fileinput.input("configs/config_tmp.py", inplace=True):
                if line.startswith("seed ="):
                    line = "seed = "+str(seed)+"\n"
                # This will replace string "a" with "truck" in each line
                    
                
                # write() method of sys module redirects the .stdout is redirected to the file
                sys.stdout.write(line)
            from train import train
            set_deterministic(seed)
            train()
            sample(experiment=[file_name[7:-3]+"_"+str(seed)])


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
    import importlib
    import sys
    # set_deterministic(args.seed)
    #print(args)
    # if args.mode == "train":
    #     train(args)

    # if args.mode == "eval":
    #     #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    #     from metrics import evaluation

    #     evaluation()

    # if args.mode == "sample":
    #     from sample import sample
    #     sample()

    # if args.mode == "paper":
    exp_names = ['config_qwtargan_best4.py',
                'config_qwqtargan_best4.py', 
                'config_qwtargan.py', 
                'config_wqtargan.py', 
                'config_targan.py', 
                'config_qtargan.py', 
                'config_qwqtargan.py', 
                'config_wtargan.py'
                ]
    paper(exp_names)

