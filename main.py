import torch
import random
import numpy as np
import json
from munch import Munch
def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False

#device = torch.device("cuda:1,3" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.


if __name__ == '__main__':
    f = open('config.json',)

    # returns JSON object as
    # a dictionary
    data = json.load(f)
    args = Munch(data)
    args["real"] = False if args["real"] =="False" else True
    args["soup"] = False if args["soup"] =="False" else True
    args["qsn"] = False if args["qsn"] =="False" else True
    args["real"]= False if args["real"] =="False" else True
    args["phm"]= False if args["phm"] =="False" else True
    args["last_layer_gen_real"]= False if args["last_layer_gen_real"] =="False" else True
    args["share_net_real"]= False if args["share_net_real"] =="False" else True
    args["betas"]= eval(args["betas"])
    args["modals"]= eval(args["modals"])

    device = torch.device('cuda:'+str(args.gpu_num) if torch.cuda.is_available() else 'cpu')

    set_deterministic(args.seed)
