import argparse
import os
from sys import int_info
"""
Detailed hyper-parameter configurations.
"""
class Param:

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser = self.all_param(parser)
        all_args, unknown = parser.parse_known_args()
        self.args = all_args

    def all_param(self, parser):

        ##################################common parameters####################################
        parser.add_argument("--gpu", default=0, type=int)

        parser.add_argument("--dataname", default='FewRel', type=str, help="Use TACRED or FewRel datasets.")

        parser.add_argument("--task_name", default='FewRel', type=str)

        parser.add_argument("--max_length", default=256, type=int)

        parser.add_argument("--this_name", default="continual", type=str)

        parser.add_argument("--device", default="cuda", type=str)

        parser.add_argument("--denoise", action="store_true", default=False, help="denoise")
        parser.add_argument("--attack", action="store_true", default=False, help="attack")
        parser.add_argument("--hidden", action="store_true", default=False, help="if output all hiddens")

        parser.add_argument("--thresh", default=0.5, type=float, help="loss threshold to separate clean and noisy samples")

        ###############################   training ################################################

        parser.add_argument("--batch_size", default=16, type=int)

        parser.add_argument("--learning_rate", default=1e-5, type=float)

        parser.add_argument("--lr2", default=3e-5, type=float)
        
        parser.add_argument("--total_round", default=5, type=int)
        
        parser.add_argument("--rel_per_task", default=8)

        parser.add_argument("--pattern", default="entity_marker") 

        parser.add_argument("--encoder_output_size", default=768, type=int)

        parser.add_argument("--vocab_size", default=30522, type =int)

        parser.add_argument("--marker_size", default=4, type=int)

        parser.add_argument("--log_path", default="logs/", type=str)

        # Temperature parameter in CL and CR
        parser.add_argument("--temp", default=0.1, type=float)
        # margin for amc
        parser.add_argument("--margin", default=0.2, type=float)
        parser.add_argument("--amc", action="store_true", default=False)

        # The projection head outputs dimensions
        parser.add_argument("--feat_dim", default=64, type=int)

        # Temperature parameter in KL
        parser.add_argument("--kl_temp", default=10, type=float)

        parser.add_argument("--num_workers", default=0, type=int)

        # epoch1
        parser.add_argument("--step1_epochs", default=1, type=int) 

        # epoch2
        parser.add_argument("--step2_epochs", default=1, type=int) 
        parser.add_argument("--split_steps", default=6, type=int) 

        parser.add_argument("--seed", default=2021, type=int) 

        parser.add_argument("--max_grad_norm", default=10, type=float) 

        # noise rate
        parser.add_argument("--noise_rate", default=0.3, type=float) 
        # Memory size
        parser.add_argument("--num_protos", default=20, type=int)

        parser.add_argument("--optim", default='adamw', type=str)

        # dataset path
        parser.add_argument("--data_path", default='data/', type=str)

        # bert-base-uncased weights path
        parser.add_argument("--bert_path", default="/root/bert-base-uncased", type=str)
        
        return parser