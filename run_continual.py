import os
import torch
from torch.utils.tensorboard import SummaryWriter

from config import Param
from methods.utils import setup_seed
from methods.manager import Replay_Manager
from methods.noise_free_manager import Noise_Free_Manager

def run(args, tensor_writer):
    setup_seed(args.seed)
    print("hyper-parameter configurations:")
    print(str(args.__dict__))
    
    manager = Noise_Free_Manager(args, tensor_writer)
    manager.train(args)


if __name__ == '__main__':
    param = Param() # There are detailed hyper-parameter configurations.
    args = param.args
    torch.cuda.set_device(args.gpu)
    args.device = torch.device(args.device)
    args.n_gpu = torch.cuda.device_count()
    args.task_name = args.dataname
    args.rel_per_task = 8 if args.dataname == 'FewRel' else 4 
    if args.dataname == 'FewRel':
        args.num_of_test = 140
    if args.dataname == 'Tacred':
        args.data_file = 'data/data_with_marker_tacred.json' 
        args.relation_file = 'data/id2rel_tacred.json'
        args.num_of_relation = 40

    #tensor_writer = SummaryWriter(args.log_path)
    tensor_writer = None
    run(args, tensor_writer)
    

   
