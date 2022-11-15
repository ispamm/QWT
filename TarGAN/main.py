import os
import munch

import torch
import random
import numpy as np

# from dataset import ChaosDataset_Syn_new
# # from metrics import compute_miou, create_images_for_dice_or_s_score, evaluate, evaluation, calculate_ignite_inception_score, png_series_reader
# from utils import build_model, load_nets
from tqdm import tqdm
import shutil
import argparse
#from sample import sample
def paper(exp_names):
    seeds = [1761017,1704899, 1245135, 2058486, 123152352]
    for i in range(3):
        seed = seeds[i] 
        for file_name in tqdm(exp_names, total=len(exp_names)):
            file_path_original = os.path.join("configs",file_name)
            # module_name = "configs."+file_name[:-3]
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
            nets = train()
            sample(nets, experiment=[file_name[7:-3]+"_"+str(seed)])


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
    from config import args

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mode', type=str, default="train",help='an integer for the accumulator')
    parser.add_argument('--seed', type=int, default=42,help='an integer for the accumulator')
    parser.add_argument('--experiment_name', type=str, default="qwtargan_novel_best_",help='an integer for the accumulator')
    parser.add_argument('--best_4', nargs='+', help='<Required> Set flag', default=0)
    args_parsed = parser.parse_args()
    args.seed = args_parsed.seed
    args.best_4 = list(map(int, args_parsed.best_4))
    args.mode= args_parsed.mode
    args.experiment_name = args_parsed.experiment_name+str(len(args.best_4))+"_moe_"+str(args.seed) #if "TEST" in args_parsed.experiment_name else args.experiment_name
    if args.mode =='eval' and args.sepoch==0:
        args.sepoch=50
    set_deterministic(args.seed)
    print(args)
    if args.mode == "train":
        from train import train
        train(args)

    if args.mode == "eval":
        from metrics import evaluation
        evaluation()

    if args.mode == "sample":
        from sample import sample
        sample()

    if args.mode == "paper":
        exp_names = [
                    'config_qwqtargan.py', 
                    'config_wtargan.py',
                    'config_qtargan.py', 
                    'config_targan.py',
                    'config_qwtargan.py',
                    'config_qwqtargan_best4.py',
                    'config_qwtargan_best4.py',
                    'config_wqtargan.py',
                    'config_qwqtargan_best4_moe.py',
                    'config_qwtargan_best4_moe.py',
                    ]
        paper(exp_names)

