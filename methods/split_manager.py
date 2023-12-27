import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from numpy import *
import random
from tqdm import tqdm, trange
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import math
import os

from dataloaders.sampler import data_sampler
from dataloaders.data_loader import get_data_loader
from .model import Encoder,BasicBertModel
from .utils import Moment,  dot_dist, osdist

class Attack_Manager(object):
    def __init__(self, args, tensor_writer):
        super().__init__()
        self.id2rel = None
        self.rel2id = None
        self.tensor_writer = tensor_writer

    def get_optimizer(self, args, encoder):
        print('Use {} optim!'.format(args.optim))
        def set_param(module, lr, decay=0):
            parameters_to_optimize = list(module.named_parameters())
            no_decay = ['undecay']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr},
                {'params': [p for n, p in parameters_to_optimize
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr}
            ]
            return parameters_to_optimize
        params = set_param(encoder, args.learning_rate)

        if args.optim == 'adam':
            pytorch_optim = optim.Adam
        elif args.optim == 'adamw':
            pytorch_optim = optim.AdamW
        else:
            raise NotImplementedError
        optimizer = pytorch_optim(
            params
        )
        return optimizer
    
    def get_optimizer_v2(self, args, encoder, is_bertcls= False):
        print('Use {} optim!'.format(args.optim))
        def set_param(module, lr, decay=0.01):
            parameters_to_optimize = list(module.named_parameters())
            no_decay = ["bias", "LayerNorm.weight"]
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize
                            if not any(nd in n for nd in no_decay)], 'weight_decay': decay, 'lr': lr},
                {'params': [p for n, p in parameters_to_optimize
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr}
            ]
            return parameters_to_optimize
        if is_bertcls is False:
            params = set_param(encoder, args.learning_rate)
        else:
            params = set_param(encoder, args.lr2)
        if args.optim == 'adam':
            pytorch_optim = optim.Adam
        elif args.optim == 'adamw':
            pytorch_optim = optim.AdamW
        else:
            raise NotImplementedError
        optimizer = pytorch_optim(
            params
        )
        return optimizer

    def train_bert_cls(self, args, bert_cls, training_data, epochs, task_num, tmp_lb2idx, relation_cnt_dict):
        data_loader = get_data_loader(args, training_data, shuffle=True)
        optimizer = self.get_optimizer_v2(args, bert_cls, is_bertcls=True)

        def train_data(data_loader_, name = "", total_step=0, task_num=1, is_mem = False):
            bert_cls.train()
            losses = []
            td = tqdm(data_loader_, desc=name)
            for step, batch_data in enumerate(td):
                optimizer.zero_grad()
                labels, tokens, flag, ind = batch_data
                pseudo_labels = torch.tensor([torch.tensor(tmp_lb2idx[x.item()]) for x in labels])
                labels = labels.to(args.gpu)
                pseudo_labels = pseudo_labels.to(args.gpu)
                tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
                logits,_ = bert_cls.forward(tokens)
                #loss_reps = torch.cat((reps,aug_reps),dim=0)
                #loss_labels = torch.cat((labels,labels),dim=0)
                #print(logits.shape,pseudo_labels.shape,pseudo_labels.squeeze(-1).shape)
                loss = F.cross_entropy(logits, pseudo_labels)
                #tensor_writer.add_scalar(f'Task-{task_num} loss', loss, total_step)

                losses.append(loss.item())
                td.set_postfix(loss = np.array(losses).mean())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(bert_cls.parameters(), args.max_grad_norm)
                optimizer.step()
                total_step += 1
            print(f"{name} loss is {np.array(losses).mean()}")
        
        def split_data_v2():
            data_loader = get_data_loader(args, training_data, shuffle=False, batch_size=1)
            td = tqdm(data_loader)
            bert_cls.eval()
            cnt = 0
            cnt_true = 0
            cnt_false = 0
            clean_data = list()
            noisy_data = list()
            clean_data_dict = dict()
            predict_data_dict = dict()
            clean_loss_list = list()
            clean_confidence_list = list()
            noisy_loss_list = list()
            noisy_confidence_list = list()
            # generate long-tail distribution using model prediction
            for step, batch_data in enumerate(td):
                with torch.no_grad():
                    labels, tokens, flag, ind = batch_data
                    pseudo_labels = torch.tensor([torch.tensor(tmp_lb2idx[x.item()]) for x in labels])
                    labels = labels.to(args.gpu)
                    pseudo_labels = pseudo_labels.to(args.gpu)
                    tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
                    logits,_ = bert_cls.forward(tokens)
                    softmax_logits = F.softmax(logits,dim=-1)
                    argmax_pi = torch.argmax(softmax_logits,dim=-1)
                    relation = training_data[step]['relation']
                    if relation not in predict_data_dict.keys():
                        predict_data_dict[relation] = 0
                    if argmax_pi == pseudo_labels:
                        predict_data_dict[relation]+=1 # predict true
            all_true = 0
            for key in predict_data_dict.keys():
                all_true += predict_data_dict[key]
            for key in predict_data_dict.keys():
                predict_data_dict[key] /= all_true
            print("predict data dict...")
            print(predict_data_dict)
            # splitting data
            for step, batch_data in enumerate(td):
                with torch.no_grad():
                    labels, tokens, flag, ind = batch_data
                    pseudo_labels = torch.tensor([torch.tensor(tmp_lb2idx[x.item()]) for x in labels])
                    labels = labels.to(args.gpu)
                    pseudo_labels = pseudo_labels.to(args.gpu)
                    tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
                    logits,_ = bert_cls.forward(tokens)
                    loss = F.cross_entropy(logits, pseudo_labels)
                    softmax_logits = F.softmax(logits,dim=-1)
                    argmax_pi = torch.argmax(softmax_logits,dim=-1)
                    max_pi = torch.max(softmax_logits,dim=-1).values.item()
                    print("max pi...")
                    print(max_pi)
                    print(flag)
                    if flag.item() is True:
                        clean_confidence_list.append(max_pi)
                        clean_loss_list.append(loss.item())
                    else:
                        noisy_confidence_list.append(max_pi)
                        noisy_loss_list.append(loss.item())
            if not os.path.exists("/root/crl-new/outputs115/{}_noise{}".format(args.task_name,args.noise_rate)):
                os.mkdir("/root/crl-new/outputs115/{}_noise{}".format(args.task_name,args.noise_rate))
            np.save("/root/crl-new/outputs115/{}_noise{}/task{}-clean_loss.npy".format(args.task_name,args.noise_rate,task_num), clean_loss_list)
            np.save("/root/crl-new/outputs115/{}_noise{}/task{}-clean_confidence.npy".format(args.task_name,args.noise_rate,task_num), clean_confidence_list)
            np.save("/root/crl-new/outputs115/{}_noise{}/task{}-noisy_loss.npy".format(args.task_name,args.noise_rate,task_num), noisy_loss_list)
            np.save("/root/crl-new/outputs115/{}_noise{}/task{}-noisy_confidence.npy".format(args.task_name,args.noise_rate,task_num), noisy_confidence_list)
        total_step = 0
        for epoch_i in range(epochs):
            train_data(data_loader, "init_train_bertcls{}".format(epoch_i), total_step, task_num=task_num, is_mem=False)
            #eval_data()
        split_data_v2()

               
    def kl_div_loss(self, x1, x2):

        batch_dist = F.softmax(x1, dim=-1)
        temp_dist = F.log_softmax(x2, dim=-1)
        loss = F.kl_div(temp_dist, batch_dist, reduction="batchmean")
        loss /= 12
        print("kl loss is {}".format(loss))
        return loss

    def train(self, args):
        # set training batch
        for i in range(args.total_round):
            test_cur = []
            test_total = []
            test_acc = []
            test_forget = []
            relation_cnt = dict()
            thresh_tmp = args.thresh
            if args.noise_rate == 0.59 and i == 1:
                args.thresh = 0.55
            else:
                args.thresh = thresh_tmp
            # set random seed
            random.seed(args.seed+(i)*100)
            # sampler setup
            sampler = data_sampler(args=args, seed=args.seed+(i)*100)
            self.id2rel = sampler.id2rel
            self.rel2id = sampler.rel2id
            # encoder setup
            encoder = Encoder(args=args,hidden=args.hidden).to(args.gpu)

            # initialize memory and prototypes
            num_class = len(sampler.id2rel)
            memorized_samples = {}

            # load data and start computation
            history_relation = []
            proto4repaly = []
            for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):
                print(current_relations)
                current_labels = [self.rel2id[r] for r in current_relations]
                print(current_labels)
                tmp_lb2idx = {k:v for v,k in enumerate(current_labels)}
                tmp_idx2lb = {k:v for v,k in tmp_lb2idx.items()}
                #print(training_data)
                # reconstruct training data
                # Initial
                bert_cls = BasicBertModel(args=args,num_labels=args.rel_per_task,hidden=True).to(args.gpu) # model to seperate noisy and clean
                relation_cnt_dict = dict()
                train_data_for_initial = []
                for relation in current_relations:
                    history_relation.append(relation)
                    train_data_for_initial += training_data[relation] # turn dict into list again
                    relation_cnt_dict[self.rel2id[relation]] = len(training_data[relation])
                for relation in current_relations:
                    relation_cnt_dict[self.rel2id[relation]] /= len(train_data_for_initial)

                print("relation cnt is...")
                print(relation_cnt_dict)
                #noisy_embedding_list = self.train_bert_cls_with_noisy(args,bert_cls,train_data_for_initial,5,tmp_lb2idx,{})
                
                # getting clean data and noisy data
                self.train_bert_cls(args,bert_cls,train_data_for_initial,args.split_steps,steps+1,tmp_lb2idx,relation_cnt_dict)
                del bert_cls

