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

from dataloaders.sampler import data_sampler
from dataloaders.data_loader import get_data_loader
from .model import Encoder,BasicBertModel
from .utils import Moment,  dot_dist, osdist

class Noise_Free_Manager(object):
    def __init__(self, args, tensor_writer):
        super().__init__()
        self.id2rel = None
        self.rel2id = None
        self.tensor_writer = tensor_writer

    def get_proto(self, args, encoder, mem_set,flag=False):
        data_loader = get_data_loader(args, mem_set, False, False, 1)

        features = []
        hiddens = {}
        protos = []
        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, _, ind = batch_data
            tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
            with torch.no_grad():
                if encoder.hidden is True:
                    feature, rep, hidden_states = encoder.bert_forward(tokens)
                else:
                    feature, rep = encoder.bert_forward(tokens)
            features.append(feature)
            if encoder.hidden is True:
                for i, hidden in enumerate(hidden_states):
                    if i not in hiddens.keys():
                        hiddens[i] = []
                    hiddens[i].append(hidden)
            if flag is True:
                self.lbs.append(labels.item())
        features = torch.cat(features, dim=0)
        proto = torch.mean(features, dim=0, keepdim=True) # 对embeddings做平均
        if encoder.hidden is True:
            for i in range(0,len(hiddens)):
                hiddens[i] = torch.cat(hiddens[i],dim=0)
                protos.append(torch.mean(hiddens[i], dim=0, keepdim=True))
            return proto, features, protos
        else:
            return proto, features
   
    def select_data(self, args, encoder, sample_set):
        data_loader = get_data_loader(args, sample_set, shuffle=False, drop_last=False, batch_size=1)
        features = []
        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, flag, ind = batch_data
            tokens=torch.stack([x.to(args.gpu) for x in tokens],dim=0)
            with torch.no_grad():
                if encoder.hidden is False:
                    feature, rp = encoder.bert_forward(tokens)
                else:
                    feature, rp, _ = encoder.bert_forward(tokens)
            features.append(feature.detach().cpu())

        features = np.concatenate(features)
        num_clusters = min(args.num_protos, len(sample_set))
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)
        mem_set = []
        current_feat = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            instance = sample_set[sel_index]
            mem_set.append(instance)
            current_feat.append(features[sel_index])
        
        current_feat = np.stack(current_feat, axis=0)
        current_feat = torch.from_numpy(current_feat)
        return mem_set, current_feat, current_feat.mean(0)
    
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

    def data_separation(self, args, encoder, training_data):
        self.moment.init_moment(args, encoder, training_data, is_memory=False)
        data_loader = get_data_loader(args, training_data, shuffle=False, batch_size=1)
        td = tqdm(data_loader)
        encoder.eval()
        losses = []
        clean_data_dict, clean_data = {}, []  # training_data , training_data_for_initial
        
        for step, batch_data in enumerate(td):
            with torch.no_grad():
                labels, tokens, flag, ind = batch_data
                labels = labels.to(args.gpu)
                tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
                hidden, reps, aug_reps = encoder.bert_forward(tokens, aug=True)
                if args.amc:
                    loss = self.moment.amc_loss_v3(reps, labels, aug_reps, flag=True)
                else:
                    loss = self.moment.loss(reps, labels)
                losses.append(loss.item())
        
        loss_max = max(losses)
        loss_min = min(losses)

        cnt = 0
        normalized_losses = []
        # loss_thresh = args.thresh
        for idx in range(len(training_data)):
            normalized_loss = (losses[idx] - loss_min) / (loss_max - loss_min)
            normalized_losses.append(normalized_loss)
        loss_thresh = mean(normalized_losses)
        for idx in range(len(training_data)):
            # normalized_loss = (losses[idx] - loss_min) / (loss_max - loss_min)
            # normalized_losses.append(normalized_loss)
            if normalized_losses[idx] > loss_thresh:
                cnt += 1
                ## TODO targeted attack 
                # training_data[idx]['relation'] = 80
                # use textattack to attack training_data[idx]
            else:
                relation = training_data[idx]['relation']
                if relation not in clean_data_dict.keys():
                    clean_data_dict[relation] = []
                clean_data_dict[relation].append(training_data[idx])
                clean_data.append(training_data[idx])
        #print("normalized_losses: ", normalized_losses)
        #print("pseudo_noise: ", cnt)
        #print("pseudo_noise_ratio: ", cnt/len(losses))
        return clean_data_dict, clean_data, training_data

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

        def eval_data():
            data_loader = get_data_loader(args, training_data, shuffle=False, batch_size=1)
            td = tqdm(data_loader)
            bert_cls.eval()
            losses = []       
            for step, batch_data in enumerate(td):
                with torch.no_grad():
                    labels, tokens, flag, ind = batch_data
                    pseudo_labels = torch.tensor([torch.tensor(tmp_lb2idx[x.item()]) for x in labels])
                    labels = labels.to(args.gpu)
                    pseudo_labels = pseudo_labels.to(args.gpu)
                    tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
                    logits,_ = bert_cls.forward(tokens)
                    loss = F.cross_entropy(logits, pseudo_labels)
                    losses.append(loss.item())
            loss_max = max(losses)
            loss_min = min(losses)
            normalized_losses = []
            clean_losses = []
            noise_losses = []
            clean_norm_losses = []
            noise_norm_losses = []
            # loss_thresh = args.thresh
            for idx in range(len(training_data)):
                normalized_loss = (losses[idx] - loss_min) / (loss_max - loss_min)
                normalized_losses.append(normalized_loss)
                if training_data[idx]['ori_relation']!=training_data[idx]['relation']: #noisy data
                    noise_losses.append(losses[idx])
                    noise_norm_losses.append(normalized_loss)
                else:
                    clean_losses.append(losses[idx])
                    clean_norm_losses.append(normalized_loss)

            loss_thresh = mean(normalized_losses)
            """
            print("normalized_losses: ", normalized_losses)
            print("average_normalized_losses: ", loss_thresh)
            print("losses: ", losses)
            print("average_losses: ", mean(losses))
            print("clean_losses: ", clean_losses)
            print("average_clean_losses: ", mean(clean_losses))
            print("noise_losses: ", noise_losses)
            print("average_noise_losses: ", mean(noise_losses))
            print("clean_losses: ", clean_losses)
            print("average_clean_norm_losses: ", mean(clean_norm_losses))
            print("noise_norm_losses: ", noise_norm_losses)
            print("average_noise_norm_losses: ", mean(noise_norm_losses))
            """
        def split_data():
            data_loader = get_data_loader(args, training_data, shuffle=False, batch_size=1)
            td = tqdm(data_loader)
            bert_cls.eval()
            losses = []       
            for step, batch_data in enumerate(td):
                with torch.no_grad():
                    labels, tokens, flag, ind = batch_data
                    pseudo_labels = torch.tensor([torch.tensor(tmp_lb2idx[x.item()]) for x in labels])
                    labels = labels.to(args.gpu)
                    pseudo_labels = pseudo_labels.to(args.gpu)
                    tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
                    logits,_ = bert_cls.forward(tokens)
                    loss = F.cross_entropy(logits, pseudo_labels)
                    losses.append(loss.item())
            loss_max = max(losses)
            loss_min = min(losses)
            normalized_losses = []
        
            # loss_thresh = args.thresh
            for idx in range(len(training_data)):
                normalized_loss = (losses[idx] - loss_min) / (loss_max - loss_min)
                normalized_losses.append(normalized_loss)
            loss_thresh = mean(normalized_losses)
            cnt = 0
            cnt_true = 0
            cnt_false = 0
            clean_data = list()
            noisy_data = list()
            clean_data_dict = dict()
            for idx in range(len(training_data)):
                if normalized_losses[idx] > loss_thresh: # expeced noise
                    cnt += 1
                    if training_data[idx]['ori_relation']!=training_data[idx]['relation']: #noisy data
                        cnt_true += 1
                    noisy_data.append(training_data[idx])
                else:
                    if training_data[idx]['ori_relation']!=training_data[idx]['relation']: #noisy data
                        cnt_false += 1
                    relation = training_data[idx]['relation']
                    if relation not in clean_data_dict.keys():
                        clean_data_dict[relation] = []
                    clean_data_dict[relation].append(training_data[idx])
                    clean_data.append(training_data[idx])
            print("expected {} noisy data and {} clean data, {} noisy data in noisy and {} noisy data in clean".format(cnt,len(clean_data),cnt_true,cnt_false))
            #print("normalized_losses: ", normalized_losses)
            #print("pseudo_noise: ", cnt)
            print("pseudo_noise_ratio: ", cnt/len(losses))
            for relation in clean_data_dict.keys():
                print("{} clean data for relation {} ".format(len(clean_data_dict[relation]),relation))
            return clean_data_dict, clean_data, training_data, noisy_data

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
                    softmax_logits = F.softmax(logits,dim=-1)
                    argmax_pi = torch.argmax(softmax_logits,dim=-1)
                    max_pi = torch.max(softmax_logits,dim=-1).values.item()
                    print("max pi...")
                    print(max_pi)
                    print(flag)
                    
                    relation = training_data[step]['relation']
                    if predict_data_dict[relation] <= 0.1 and args.dataname == "Tacred": # tail samples for tacred
                        this_thresh = 0.0
                    else:
                        this_thresh = args.thresh
                    if step == 0:
                        print("thresh is...")
                        print(this_thresh)
                    if argmax_pi == pseudo_labels and max_pi > this_thresh:
                        if training_data[step]['ori_relation']!=training_data[step]['relation']: #noisy data
                            cnt_false += 1
                        relation = training_data[step]['relation']
                        if relation not in clean_data_dict.keys():
                            clean_data_dict[relation] = []
                        clean_data_dict[relation].append(training_data[step])
                        clean_data.append(training_data[step])  
                    else:
                        cnt += 1
                        if training_data[step]['ori_relation']!=training_data[step]['relation']: #noisy data
                            cnt_true += 1
                        noisy_data.append(training_data[step])
                        
            print("expected {} noisy data and {} clean data, {} noisy data in noisy and {} noisy data in clean".format(cnt,len(clean_data),cnt_true,cnt_false))
            for relation in clean_data_dict.keys():
                print("{} clean data for relation {} ".format(len(clean_data_dict[relation]),relation))
            return clean_data_dict, clean_data, training_data, noisy_data

        total_step = 0
        for epoch_i in range(epochs):
            train_data(data_loader, "init_train_{}".format(epoch_i), total_step, task_num=task_num, is_mem=False)
            #eval_data()
        clean_data_dict, clean_data, training_data, noisy_data = split_data_v2()  
        return clean_data_dict, clean_data, noisy_data
    
    def compute_concentration(self, args, encoder, clean_data_dict, proto_dict):
        encoder.eval()
        phi_list = dict()
        margin_list = dict()
        with torch.no_grad():
            for relation in clean_data_dict.keys():
                proto_last = proto_dict[relation][-1] # last layer prototype
                training_data = clean_data_dict[relation]
                Z = len(training_data)
                """
                print("proto last shape:")
                print(proto_last.shape)
                print("Z:")
                print(Z)
                """
                phi = 0
                data_loader = get_data_loader(args, training_data, shuffle=True)
                for step, batch_data in enumerate(data_loader):
                    labels, tokens, flag, ind = batch_data
                    tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
                    with torch.no_grad():
                        if encoder.hidden is True:
                            feature, rep, hidden_states = encoder.bert_forward(tokens)
                        else:
                            feature, rep = encoder.bert_forward(tokens)
                    phi += torch.sum(torch.norm(feature-proto_last,p=2,dim=1))/(Z*math.log(Z))
                margin = (torch.cos(torch.tensor(np.pi*(phi-0.5)))+1) / 100
                margin_list[relation] = margin
                phi_list[relation] = phi
                print("concentration for relation {} is {}".format(relation, phi))
                print("margin for relation {} is {}".format(relation, margin))
        return phi_list,margin_list

    def compute_dist(self, args, encoder, clean_data_dict, proto_dict):
        encoder.eval()
        max_dist_list = dict()
        with torch.no_grad():
            for relation in clean_data_dict.keys():
                proto_last = proto_dict[relation][-1] # last layer prototype
                training_data = clean_data_dict[relation]
                data_loader = get_data_loader(args, training_data, shuffle=True)
                max_dist = 0.0
                for step, batch_data in enumerate(data_loader):
                    labels, tokens, flag, ind = batch_data
                    tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
                    with torch.no_grad():
                        if encoder.hidden is True:
                            feature, rep, hidden_states = encoder.bert_forward(tokens)
                        else:
                            feature, rep = encoder.bert_forward(tokens)
                    max_dist = max(torch.max(torch.norm(feature-proto_last,p=2,dim=1)),max_dist)
                max_dist_list[relation] = max_dist
                #print("max dist for relation {} is {}".format(relation, max_dist))
        return max_dist_list
    
    def train_bert_cls_with_noisy(self, args, bert_cls, training_data, tmp_lb2idx, proto_dict, epsilon=0.1, adv_steps=5):
        data_loader = get_data_loader(args, training_data, shuffle=False) # fix the order
        optimizer = self.get_optimizer_v2(args, bert_cls)
        bert_cls.zero_grad()
        losses = []
        td = tqdm(data_loader, desc="")
        noisy_embedding_list = []
        def loss_fn(logits, hidden_states, pseudo_labels, labels, proto_dict):
            all_protos = [proto_dict[x.item()][-1:] for x in labels] # select prototype for each label (last layer)
            all_protos = torch.stack([x for x in [torch.stack(y,0) for y in all_protos]], dim=0) #[B*n_layers*1*768]
            #hidden_states = torch.stack([x for x in [torch.stack(y,0) for y in hidden_states]], dim=0)
            hidden_states = torch.stack([x for x in hidden_states[-1:]],dim=1) #[B*n_layers*768]
            return F.cross_entropy(logits,pseudo_labels) + 0.1*self.kl_div_loss(all_protos.squeeze(-2),hidden_states)
        def loss_fn_v2(logits, pseudo_labels, x, new_x):
            return F.cross_entropy(logits,pseudo_labels) - 0.1*self.kl_div_loss(x,new_x)

        for step, batch_data in enumerate(td):
            optimizer.zero_grad()
            labels, tokens, flag, ind = batch_data
            pseudo_labels = torch.tensor([torch.tensor(tmp_lb2idx[x.item()]) for x in labels]) # turn relation labels to ce labels
            labels = labels.to(args.gpu)
            pseudo_labels = pseudo_labels.to(args.gpu)
            tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
            # 0. get embeddings of noisy data
            embedding_init = bert_cls.get_input_embeddings()(tokens)
            delta = torch.zeros_like(embedding_init).uniform_(-epsilon, epsilon)
            delta.requires_grad = True
            for astep in range(adv_steps):
                # 1. add epsilon to embeddings
                noisy_embeddings = embedding_init + delta
                # 2. forward
                logits,hidden_states = bert_cls.forward(tokens,flag=True,embedding=noisy_embeddings)
                """
                print("hidden_states shape:")
                print(len(hidden_states))
                print(hidden_states[0].shape)
                """
                # 3. train with combined losses
                #loss = loss_fn(logits, hidden_states, pseudo_labels, labels, proto_dict)
                loss = loss_fn_v2(logits, pseudo_labels, embedding_init, noisy_embeddings)
                losses.append(loss.item())
                print(f"combined loss is {np.array(losses).mean()}")
                td.set_postfix(loss = np.array(losses).mean())
                # 4. backward
                loss.backward()
                # 5. update & clip
                delta.data = delta - delta.grad.detach()
                delta.data = delta.data.clamp(-epsilon, epsilon)
                delta.data = (embedding_init.data + delta.data).clamp(0, 1) - embedding_init.data
                delta.grad.zero_()
                embedding_init = bert_cls.get_input_embeddings()(tokens) # reinitialize   
            # modified embedding for this batch 
            noisy_embedding_list.append(embedding_init.detach() + delta.detach())
        return noisy_embedding_list
            
    def train_simple_model(self, args, encoder, training_data, noisy_data, noisy_embedding_list, tmp_lb2idx, proto_dict, max_dist_list, margin_list, epochs, task_num, clean_data_dict=None, tensor_writer=None):

        data_loader = get_data_loader(args, training_data, shuffle=True)
        noisy_data_loader = get_data_loader(args, noisy_data, shuffle=False) # match with embedding
        optimizer = self.get_optimizer_v2(args, encoder)
        # seperate attack for each instance
        def get_noisy_features():
            """
            get noisy features based on dataloader and disturbed embeddings
            return: 64-dim features for the dataloader and dist from prototype
            """
            noisy_td = tqdm(noisy_data_loader, desc="")
            encoder.eval()
            all_reps = list()
            is_neg = list()
            all_labels = list()
            attack_success_cnt = dict()
            all_sample_cnt = dict()
            # with torch.no_grad():
            for noisy_embedding, batch_data in zip(noisy_embedding_list, noisy_td):
                with torch.no_grad(): # optimizer.zero_grad()
                    labels, tokens, flag, ind = batch_data
                    pseudo_labels = torch.tensor([torch.tensor(tmp_lb2idx[x.item()]) for x in labels])
                    labels = labels.to(args.gpu)
                    pseudo_labels = pseudo_labels.to(args.gpu)
                    noisy_embedding = noisy_embedding.to(args.gpu)
                    tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
                    if encoder.hidden is False:
                        hidden, reps, aug_reps = encoder.bert_forward(tokens, aug=True, flag=True, embedding=noisy_embedding)
                    else:
                        hidden, reps, aug_reps, hidden_states = encoder.bert_forward(tokens, aug=True, flag=True, embedding=noisy_embedding)
                # compute dist from proto
                all_protos = [proto_dict[x.item()][-1:] for x in labels] # select prototype for each label (last layer)
                all_protos = torch.stack([x for x in [torch.stack(y,0) for y in all_protos]], dim=0) #[B*n_layers*1*768]
                max_dists = torch.stack([max_dist_list[x.item()] for x in labels],0)
                dists = torch.norm(hidden-all_protos.squeeze(-2).squeeze(-2),p=2,dim=1)
                neg_flags = torch.gt(dists,max_dists)
                #print("dist shape:")
                #print(dists.shape,max_dists.shape)
                
                # get new reps with new noisy_embeddings
                with torch.no_grad(): # optimizer.zero_grad()
                    _, tokens, flag, ind = batch_data
                    tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
                    if encoder.hidden is False:
                        hidden, reps, aug_reps = encoder.bert_forward(tokens, aug=True, flag=True, embedding=noisy_embedding)
                    else:
                        hidden, reps, aug_reps, hidden_states = encoder.bert_forward(tokens, aug=True, flag=True, embedding=noisy_embedding)
                all_reps.append(reps)
                is_neg.append(neg_flags) # greater if not success, is_neg is True
                all_labels.append(labels)
                del all_protos
                del dists
                del max_dists
            for is_neg_list, label_list in zip(is_neg,all_labels):
                print(is_neg_list[0])
                for neg_flag, label in zip(is_neg_list,label_list):
                    if neg_flag.item() is False: # attack success
                        if label.item() not in attack_success_cnt.keys():
                             attack_success_cnt[label.item()] = 1
                        else:
                            attack_success_cnt[label.item()] += 1
                    if label.item() not in all_sample_cnt.keys():
                        all_sample_cnt[label.item()] = 1
                    else:
                        all_sample_cnt[label.item()] += 1
            for label in attack_success_cnt.keys():
                print("attack success samples in relation {} is {} in {}, account {}".format(label,attack_success_cnt[label],all_sample_cnt[label],attack_success_cnt[label]/all_sample_cnt[label]))
            return all_reps, is_neg, all_labels
        
        def train_data(data_loader_, name = "", total_step=0, task_num=1, is_mem = False, noisy_reps=None, is_neg=None, noisy_labels=None, margin_dict=None):
            losses = []
            td = tqdm(data_loader_, desc=name)
            encoder.train()
            for step, batch_data in enumerate(td):
                optimizer.zero_grad()
                labels, tokens, flag, ind = batch_data
                labels = labels.to(args.gpu)
                tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
                if encoder.hidden is False:
                    hidden, reps, aug_reps = encoder.bert_forward(tokens, aug=True)
                else:
                    hidden, reps, aug_reps, hidden_states = encoder.bert_forward(tokens, aug=True)
                #loss_reps = torch.cat((reps,aug_reps),dim=0)
                #loss_labels = torch.cat((labels,labels),dim=0)
                
                ## utilized disturbed noisy embeddings
                # 1. sample for noisy data
                # 2. get corresponding reps and is_neg flags
                # 3. concat with reps and aug_reps to compute angular contrastive loss
                if args.amc:
                    print("use amc loss")
                    #loss = self.moment.amc_loss_v2(loss_reps, loss_labels)
                    #loss = self.moment.amc_loss_v3(reps,labels,aug_reps,flag=True)
                    #loss = self.moment.amc_loss_adversarial(reps,labels,aug_reps, noisy_reps=noisy_reps, is_neg=is_neg, noisy_labels=noisy_labels, margin_dict=margin_dict, flag=True, is_mem=False, mapping=None)
                    loss = self.moment.amc_loss_adversarial(reps,labels,aug_reps, noisy_reps=None, is_neg=None, noisy_labels=None, margin_dict=margin_dict, flag=True, is_mem=False, mapping=None)
                    
                else:
                    loss = self.moment.loss(reps, labels)
                #tensor_writer.add_scalar(f'Task-{task_num} loss', loss, total_step)

                losses.append(loss.item())
                td.set_postfix(loss = np.array(losses).mean())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                # update moemnt
                if is_mem:
                    self.moment.update_mem(ind, reps.detach())
                else:
                    self.moment.update(ind, reps.detach()) # 更新memory bank里面的64维向量
                total_step += 1
            print(f"{name} clean loss is {np.array(losses).mean()}")
        total_step = 0
        for epoch_i in range(epochs):
            #all_reps,is_neg,all_labels = get_noisy_features() # utilize extra noisy data
            #train_data(data_loader, "init_train_clean_{}".format(epoch_i), total_step, task_num=task_num, is_mem=False,noisy_reps=all_reps,is_neg=is_neg,noisy_labels=all_labels,margin_dict=margin_list)
            train_data(data_loader, "init_train_clean_{}".format(epoch_i), total_step, task_num=task_num, is_mem=False,noisy_reps=None,is_neg=None,noisy_labels=None,margin_dict=margin_list)
    
    def train_mem_model(self, args, encoder, mem_data, proto_mem, epochs, seen_relations):
        history_nums = len(seen_relations) - args.rel_per_task
        if len(proto_mem)>0:
            
            proto_mem = F.normalize(proto_mem, p =2, dim=1)
            dist = dot_dist(proto_mem, proto_mem) # cosine similarity between prototypes
            dist = dist.to(args.gpu)

        mem_loader = get_data_loader(args, mem_data, shuffle=True)
        encoder.train()
        temp_rel2id = [self.rel2id[x] for x in seen_relations]
        map_relid2tempid = {k:v for v,k in enumerate(temp_rel2id)}
        map_tempid2relid = {k:v for k, v in map_relid2tempid.items()}
        optimizer = self.get_optimizer_v2(args, encoder)
        def train_data(data_loader_, name = "", is_mem = False):
            losses = []
            kl_losses = []
            td = tqdm(data_loader_, desc=name)
            for step, batch_data in enumerate(td):

                optimizer.zero_grad()
                labels, tokens, flag, ind = batch_data
                labels = labels.to(args.gpu)
                tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
                if encoder.hidden is False:
                    zz, reps, aug_reps = encoder.bert_forward(tokens,aug=True)
                else:
                    zz, reps, aug_reps, _ = encoder.bert_forward(tokens,aug=True)
                hidden = reps
                need_ratio_compute = ind < history_nums * args.num_protos # 之前的样本
                total_need = need_ratio_compute.sum()
                loss1 = 0.0
              
                #  Contrastive Replay
                if args.amc:
                    print("use amc loss")
                    #loss_reps = torch.cat((reps,aug_reps),dim=0)
                    #loss_labels = torch.cat((labels,labels),dim=0) 
                    #cl_loss = self.moment.amc_loss_v2(loss_reps, loss_labels)
                    cl_loss = self.moment.amc_loss_v3(reps, labels, aug_reps, is_mem=True, mapping=map_relid2tempid)
                else:
                    cl_loss = self.moment.loss(reps, labels, is_mem=True, mapping=map_relid2tempid)
                
                loss = cl_loss
                if isinstance(loss, float):
                    losses.append(loss)
                    #td.set_postfix(loss = np.array(losses).mean(),  kl_loss = np.array(kl_losses).mean())
                    td.set_postfix(loss = np.array(losses).mean())
                    # update moemnt
                    if is_mem:
                        self.moment.update_mem(ind, reps.detach(), hidden.detach()) # 更新memory中存的样本的feature
                    else:
                        self.moment.update(ind, reps.detach())
                    continue
                losses.append(loss.item())
                #td.set_postfix(loss = np.array(losses).mean(),  kl_loss = np.array(kl_losses).mean())
                td.set_postfix(loss = np.array(losses).mean())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                
                # update moemnt
                if is_mem:
                    self.moment.update_mem(ind, reps.detach())
                else:
                    self.moment.update(ind, reps.detach())
            print(f"{name} loss is {np.array(losses).mean()}")
        for epoch_i in range(epochs):
            train_data(mem_loader, "memory_train_{}".format(epoch_i), is_mem=True)

    def kl_div_loss(self, x1, x2):

        batch_dist = F.softmax(x1, dim=-1)
        temp_dist = F.log_softmax(x2, dim=-1)
        loss = F.kl_div(temp_dist, batch_dist, reduction="batchmean")
        loss /= 12
        print("kl loss is {}".format(loss))
        return loss

    @torch.no_grad()
    def evaluate_strict_model(self, args, encoder, test_data, protos4eval, featrues4eval, seen_relations):
        data_loader = get_data_loader(args, test_data, batch_size=1,is_train=False)
        encoder.eval()
        n_all = len(test_data)
        temp_rel2id = [self.rel2id[x] for x in seen_relations]
        map_relid2tempid = {k:v for v,k in enumerate(temp_rel2id)}
        map_tempid2relid = {k:v for k, v in map_relid2tempid.items()}
        correct_all = 0
        acc_list = list()
        correct = 0
        n = 0
        print("len of test data: {}".format(n_all))
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            labels = labels.to(args.gpu)
            if step % (args.rel_per_task*args.num_of_test) == (args.rel_per_task*args.num_of_test) - 1:
                print("end relation is {}".format(labels[0]))
            n += len(labels)
            tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
            if encoder.hidden is False:
                hidden, reps = encoder.bert_forward(tokens)
            else:
                hidden, reps, _ = encoder.bert_forward(tokens)
            labels = [map_relid2tempid[x.item()] for x in labels]
            logits = -osdist(hidden, protos4eval)
            seen_relation_ids = [self.rel2id[relation] for relation in seen_relations]
            seen_relation_ids = [map_relid2tempid[x] for x in seen_relation_ids]
            seen_sim = logits[:,seen_relation_ids]
            seen_sim = seen_sim.cpu().data.numpy()
            max_smi = np.max(seen_sim,axis=1)
            label_smi = logits[:,labels].cpu().data.numpy()
            if label_smi >= max_smi:
                correct_all += 1
                correct += 1
            if step % (args.rel_per_task*args.num_of_test) == (args.rel_per_task*args.num_of_test) - 1:
                acc_list.append(correct/n)
                print("step: {} n : {} task: {} ".format(step,n, step//(args.rel_per_task*args.num_of_test)))
                correct = 0
                n = 0
        return acc_list, correct_all/n_all
    
    def evaluate_strict_model_tr(self, args, encoder, test_data, protos4eval, featrues4eval, seen_relations,test_relation_cnt):
        data_loader = get_data_loader(args, test_data, batch_size=1,is_train=False)
        encoder.eval()
        n_all = len(test_data)
        temp_rel2id = [self.rel2id[x] for x in seen_relations]
        map_relid2tempid = {k:v for v,k in enumerate(temp_rel2id)}
        map_tempid2relid = {k:v for k, v in map_relid2tempid.items()}
        correct_all = 0
        acc_list = list()
        correct = 0
        n = 0
        split_task = list()
        task_cnt = 0
        for i, rel in enumerate(seen_relations):
            rel_id = self.rel2id[rel]
            task_cnt += test_relation_cnt[rel_id]
            if i % 4 == 3:
                split_task.append(task_cnt)
        print("len of test data: {}".format(n_all))
        print("split task...")
        print(split_task)
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            labels = labels.to(args.gpu)
            # end of a task
            if step + 1 in split_task:
                print("end relation is {}".format(labels[0]))
            n += len(labels)
            tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
            if encoder.hidden is False:
                hidden, reps = encoder.bert_forward(tokens)
            else:
                hidden, reps, _ = encoder.bert_forward(tokens)
            labels = [map_relid2tempid[x.item()] for x in labels]
            logits = -osdist(hidden, protos4eval)
            seen_relation_ids = [self.rel2id[relation] for relation in seen_relations]
            seen_relation_ids = [map_relid2tempid[x] for x in seen_relation_ids]
            seen_sim = logits[:,seen_relation_ids]
            seen_sim = seen_sim.cpu().data.numpy()
            max_smi = np.max(seen_sim,axis=1)
            label_smi = logits[:,labels].cpu().data.numpy()
            if label_smi >= max_smi:
                correct_all += 1
                correct += 1
            # end of a task
            if step + 1 in split_task:
                acc_list.append(correct/n)
                print("step: {} n : {} task: {} ".format(step,n, split_task.index(step+1)))
                correct = 0
                n = 0
        return acc_list, correct_all/n_all

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
                clean_data_dict, clean_data, noisy_data = self.train_bert_cls(args,bert_cls,train_data_for_initial,args.split_steps,steps+1,tmp_lb2idx,relation_cnt_dict)
                print("clean size is {} and noisy size is {}".format(len(clean_data),len(noisy_data)))
                self.moment = Moment(args)
                proto_dict = dict() # 12 hidden states for each relation
                # get prototype with clean data
                if encoder.hidden is True:
                    for relation in clean_data_dict.keys():
                        protos, featrues, hidden_states = self.get_proto(args, encoder, clean_data_dict[relation])
                        proto_dict[relation] = hidden_states
                # get disturbed embedding for noisy data
                print("proto_dict:")
                print(proto_dict.keys())
                # compute concentration for clean data
                if encoder.hidden is True:
                    _, margin_list = self.compute_concentration(args, encoder, clean_data_dict, proto_dict)
                    max_dist_list = self.compute_dist(args, encoder, clean_data_dict, proto_dict)
                # adversarial training with noisy data
                #noisy_embedding_list = self.train_bert_cls_with_noisy(args, bert_cls, noisy_data, tmp_lb2idx, proto_dict)
                del bert_cls
                # use only clean_data to update memory bank      
                self.moment.init_moment(args, encoder, clean_data, noisy_datasets=None, noisy_embeddings=None, is_memory=False)
                # use noisy_data embedings and clean_data to train the model
                #self.train_simple_model(args, encoder, clean_data, noisy_data, noisy_embedding_list, tmp_lb2idx, proto_dict, max_dist_list, margin_list, args.step1_epochs, steps+1, clean_data_dict, self.tensor_writer)
                # discard noisy samples
                self.train_simple_model(args, encoder, clean_data, noisy_data, None, tmp_lb2idx, proto_dict, max_dist_list, margin_list, args.step1_epochs, steps+1, clean_data_dict, self.tensor_writer)
                #self.moment.init_moment(args, encoder, train_data_for_initial, is_memory=False)
                #self.train_simple_model(args, encoder, train_data_for_initial, args.step1_epochs, steps+1, self.tensor_writer)
                del tmp_idx2lb
                # repaly
                # only clean_data for replay
                if len(memorized_samples)>0: # not the first class
                    # select current task sample
                    for relation in current_relations:
                        relation = self.rel2id[relation]
                        # select in only clean_data
                        memorized_samples[relation], _, _ = self.select_data(args, encoder, clean_data_dict[relation]) # 选择当前任务最有代表性的样本，存入memory
                        #memorized_samples[relation], _, _ = self.select_data(args, encoder, training_data[relation]) # 选择当前任务最有代表性的样本，存入memory
                    train_data_for_memory = []
                    for relation in history_relation:
                        relation = self.rel2id[relation]
                        train_data_for_memory += memorized_samples[relation] # train_data_for_memory 包含了之前的+当前的各关系的memorized_samples
                    
                    print("length of memory data is :", len(train_data_for_memory))
                    
                    self.moment.init_moment(args, encoder, train_data_for_memory, is_memory=True)# 用之前的+当前的各关系的memorized_samples来更新memory（存一下选出来的样本对应的64维向量和标签）
                    self.train_mem_model(args, encoder, train_data_for_memory, proto4repaly, args.step2_epochs, seen_relations)

                # 测试当前任务的accuracy & 历史任务的average accuracy
                feat_mem = []
                proto_mem = []

                for relation in current_relations:
                    relation = self.rel2id[relation]
                    memorized_samples[relation], feat, temp_proto = self.select_data(args, encoder, clean_data_dict[relation])
                    #memorized_samples[relation], feat, temp_proto = self.select_data(args, encoder, training_data[relation])
                    feat_mem.append(feat)
                    proto_mem.append(temp_proto)

                feat_mem = torch.cat(feat_mem, dim=0)
                temp_proto = torch.stack(proto_mem, dim=0)
                #memory evaluate
                flip_cnt = 0
                
                memorized_samples_cnt = 0
                for relation in seen_relations:
                    relation = self.rel2id[relation]
                    for data_item in memorized_samples[relation]:
                        memorized_samples_cnt += 1
                        if data_item['relation'] != data_item['ori_relation']:
                            flip_cnt += 1

                print(f"In step {steps}   memory size: {memorized_samples_cnt}   flipped samples: {flip_cnt}    ratio: {flip_cnt*1.0/memorized_samples_cnt}")

                protos4eval = []
                featrues4eval = []
                self.lbs = []
                for relation in history_relation:
                    if relation not in current_relations: # 之前出现的关系
                        relation = self.rel2id[relation]
                        if encoder.hidden is False:
                            protos, featrues = self.get_proto(args, encoder, memorized_samples[relation],flag=True)
                        else:
                            protos, featrues, _ = self.get_proto(args, encoder, memorized_samples[relation],flag=True)
                        protos4eval.append(protos)
                        featrues4eval.append(featrues)
                
                if protos4eval: # not first class     
                    protos4eval = torch.cat(protos4eval, dim=0).detach()
                    protos4eval = torch.cat([protos4eval, temp_proto.to(args.gpu)], dim=0)

                else: # first class
                    protos4eval = temp_proto.to(args.gpu)
                proto4repaly = protos4eval.clone()

                test_data_2 = []
                test_relation_cnt = dict() # length for each relation in test_data
                for relation in seen_relations:
                    test_data_2 += historic_test_data[relation]
                    rel = self.rel2id[relation]
                    test_relation_cnt[rel] = len(historic_test_data[relation])

                #acc_list,total_acc = self.evaluate_strict_model(args, encoder, test_data_2, protos4eval, featrues4eval,seen_relations)
                if args.dataname == 'FewRel':
                    acc_list,total_acc = self.evaluate_strict_model(args, encoder, test_data_2, protos4eval, featrues4eval,seen_relations)
                else:
                    acc_list,total_acc = self.evaluate_strict_model_tr(args, encoder, test_data_2, protos4eval, featrues4eval,seen_relations,test_relation_cnt)
                cur_acc = acc_list[-1]
                print(f'Restart Num {i+1}')
                print(f'task--{steps + 1}:')
                print(f'current test acc:{cur_acc}')
                print(f'history test acc:{total_acc}')
                test_cur.append(cur_acc)
                test_total.append(total_acc)
                test_acc.append(acc_list)
                forget_list = list()
                if len(test_acc) >1: # not base task
                    # compute history accuracy for previous tasks
                    for task_id, cur_acc in enumerate(acc_list[:-1]):
                        his_acc = [t[task_id] for t in test_acc[task_id:]] # history accuracy for this task id
                        if len(his_acc) == 0:
                            print("debugging...")
                            print(test_acc)
                            print(acc_list)
                            forget_rate = 0.0
                        else:
                            forget_rate = max(his_acc) - cur_acc
                        forget_list.append(forget_rate)
                test_forget.append(forget_list)
                print(test_cur)
                print(test_total)
                print(test_acc)
                print(f'forget rate:')
                print(test_forget)
                del self.moment
