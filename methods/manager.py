import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm, trange
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from dataloaders.sampler import data_sampler
from dataloaders.data_loader import get_data_loader
from .model import Encoder
from .utils import Moment, dot_dist, osdist

class Replay_Manager(object):
    def __init__(self, args, tensor_writer):
        super().__init__()
        self.id2rel = None
        self.rel2id = None
        self.topk = 5
        self.tensor_writer = tensor_writer
        
    def get_proto(self, args, encoder, mem_set):
        # aggregate the prototype set for further use.
        data_loader = get_data_loader(args, mem_set, False, False, 1)

        features = []

        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, flag, ind = batch_data
            tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
            with torch.no_grad():
                feature, rep= encoder.bert_forward(tokens)
            features.append(feature)
            self.lbs.append(labels.item())
        features = torch.cat(features, dim=0)

        proto = torch.mean(features, dim=0, keepdim=True) # 对embeddings做平均

        return proto, features
    def get_proto_64(self, args, encoder, mem_set):
        # aggregate the prototype set for further use.
        data_loader = get_data_loader(args, mem_set, False, False, 1)
        reps = []
        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, flag, ind = batch_data
            tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
            with torch.no_grad():
                feature, rep= encoder.bert_forward(tokens)
            reps.append(rep)
        reps = torch.cat(reps, dim=0)

        proto = torch.mean(reps, dim=0, keepdim=True) # 对64维特征做平均

        return proto

    def get_hidden(self, args, encoder, training_set):
        # get hidden features(768 dim) for training samples.
        data_loader = get_data_loader(args, training_set)
        hiddens = []
        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, flag, ind = batch_data
            tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
            with torch.no_grad():
                hidden, _= encoder.bert_forward(tokens)
            hiddens.append(hidden)
        hiddens = torch.cat(hiddens, dim=0)
        return hiddens

    def get_hidden_64(self, args, encoder, training_set):
        # get features(64 dim) for training samples.
        data_loader = get_data_loader(args, training_set)
        reps = []
        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, flag, ind = batch_data
            tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
            with torch.no_grad():
                hidden, rep = encoder.bert_forward(tokens)
            reps.append(rep)
        reps = torch.cat(reps, dim=0)
        return reps

    def compute_proto_dist(self, args, encoder, steps, old_protos, train_data, idex2relation, seen=False):
        train_data_hidden = self.get_hidden(args, encoder, train_data)
        old_dist = cosine_similarity(train_data_hidden.cpu(),old_protos.cpu()) # numpy array
        print(old_dist.shape)
        if seen is False:
            tag = "unseen"
        else:
            tag = "seen"
        sorted_indexes = np.argsort(old_dist)
        sorted_dists = np.sort(old_dist)
        min_indexes = sorted_indexes[:,:self.topk]
        min_dists = sorted_dists[:,:self.topk]
        max_indexes = np.flip(sorted_indexes,axis=1)[:,:self.topk]
        max_dists = np.flip(sorted_dists,axis=1)[:,:self.topk]
        print("max 5 indexes:")
        print(max_indexes.shape)
        for idx,idx_list in enumerate(max_indexes):
            max_relation = [idex2relation[int(item)] for item in idx_list]
            print("find relation")
            print(max_relation)
        #np.save("/root/crl-new/outputs768/noise{}/task{}-{}-old_dist.npy".format(args.noise_rate,steps,tag), old_dist)
        #np.save("/root/crl-new/outputs768/noise{}/task{}-{}-max_dists.npy".format(args.noise_rate,steps,tag), max_dists)
        #np.save("/root/crl-new/outputs768/noise{}/task{}-{}-min_dists.npy".format(args.noise_rate,steps,tag), min_dists)
        #np.save("/root/crl-new/outputs768/noise{}/task{}-{}-max_indexes.npy".format(args.noise_rate,steps,tag), max_indexes)
        #np.save("/root/crl-new/outputs768/noise{}/task{}-{}-min_indexes.npy".format(args.noise_rate,steps,tag), min_indexes)
        #print("save .npy done")
    # Use K-Means to select what samples to save, similar to at_least = 0
    def select_data(self, args, encoder, sample_set):
        data_loader = get_data_loader(args, sample_set, shuffle=False, drop_last=False, batch_size=1)
        features = []
        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, flag, ind = batch_data
            tokens=torch.stack([x.to(args.gpu) for x in tokens],dim=0)
            with torch.no_grad():
                feature, rp = encoder.bert_forward(tokens)
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
        else:
            raise NotImplementedError
        optimizer = pytorch_optim(
            params
        )
        return optimizer
    def train_simple_model(self, args, encoder, training_data, epochs, task_num, tensor_writer=None):

        data_loader = get_data_loader(args, training_data, shuffle=True)
        encoder.train()
        optimizer = self.get_optimizer(args, encoder)

        def train_data(data_loader_, name = "", total_step=0, task_num=1, is_mem = False):
            losses = []
            td = tqdm(data_loader_, desc=name)
            for step, batch_data in enumerate(td):
                optimizer.zero_grad()
                labels, tokens, flag, ind = batch_data
                labels = labels.to(args.gpu)
                tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
                hidden, reps, aug_reps = encoder.bert_forward(tokens, aug=True)
                #loss_reps = torch.cat((reps,aug_reps),dim=0)
                #loss_labels = torch.cat((labels,labels),dim=0)
                if args.amc:
                    print("use amc loss")
                    #loss = self.moment.amc_loss_v2(loss_reps, loss_labels)
                    loss = self.moment.amc_loss_v3(reps,labels,aug_reps,flag=True)
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
            print(f"{name} loss is {np.array(losses).mean()}")

        total_step = 0
        for epoch_i in range(epochs):
            train_data(data_loader, "init_train_{}".format(epoch_i), total_step, task_num=task_num, is_mem=False)
        
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
        optimizer = self.get_optimizer(args, encoder)
        def train_data(data_loader_, name = "", is_mem = False):
            losses = []
            kl_losses = []
            td = tqdm(data_loader_, desc=name)
            for step, batch_data in enumerate(td):

                optimizer.zero_grad()
                labels, tokens, flag, ind = batch_data
                labels = labels.to(args.gpu)
                tokens = torch.stack([x.to(args.gpu) for x in tokens], dim=0)
                zz, reps, aug_reps = encoder.bert_forward(tokens,aug=True)
                hidden = reps
                need_ratio_compute = ind < history_nums * args.num_protos # 之前的样本
                total_need = need_ratio_compute.sum()
                loss1 = 0.0
                """
                if total_need >0 :
                    # Knowledge Distillation for Relieve Forgetting
                    need_ind = ind[need_ratio_compute]
                    need_labels = labels[need_ratio_compute]
                    temp_labels = [map_relid2tempid[x.item()] for x in need_labels]
                    gold_dist = dist[temp_labels]
                    current_proto = self.moment.get_mem_proto()[:history_nums]
                    this_dist = dot_dist(hidden[need_ratio_compute], current_proto.to(args.gpu))
                    loss1 = self.kl_div_loss(gold_dist, this_dist, t=args.kl_temp)
                    loss1.backward(retain_graph=True)
                else:
                    loss1 = 0.0
                """
                #  Contrastive Replay
                if args.amc:
                    print("use amc loss")
                    #loss_reps = torch.cat((reps,aug_reps),dim=0)
                    #loss_labels = torch.cat((labels,labels),dim=0) 
                    #cl_loss = self.moment.amc_loss_v2(loss_reps, loss_labels)
                    cl_loss = self.moment.amc_loss_v3(reps, labels, aug_reps, is_mem=True, mapping=map_relid2tempid)
                else:
                    cl_loss = self.moment.loss(reps, labels, is_mem=True, mapping=map_relid2tempid)
                
                """
                if isinstance(loss1, float):
                    kl_losses.append(loss1)
                else:
                    kl_losses.append(loss1.item())
                """
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
    def kl_div_loss(self, x1, x2, t=10):

        batch_dist = F.softmax(t * x1, dim=1)
        temp_dist = F.log_softmax(t * x2, dim=1)
        loss = F.kl_div(temp_dist, batch_dist, reduction="batchmean")
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
            hidden, reps = encoder.bert_forward(tokens)
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

    def train(self, args):
        # set training batch
        for i in range(args.total_round):
            test_cur = []
            test_total = []
            test_acc = []
            test_forget = []
            # set random seed
            random.seed(args.seed+i*100)

            # sampler setup
            sampler = data_sampler(args=args, seed=args.seed+i*100)
            self.id2rel = sampler.id2rel
            self.rel2id = sampler.rel2id
            # encoder setup
            encoder = Encoder(args=args).to(args.gpu)

            # initialize memory and prototypes
            num_class = len(sampler.id2rel)
            memorized_samples = {}

            # load data and start computation
            history_relation = []
            proto4repaly = []
            for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):
                print(current_relations)
                print([self.rel2id[r] for r in current_relations])
                
                #print(training_data)
                # reconstruct training data
                # Initial
                train_data_for_initial = []
                for relation in current_relations:
                    history_relation.append(relation)
                    train_data_for_initial += training_data[relation] # turn dict into list again

                # train model
                # no memory. first train with current task / init trainging for new task
                self.moment = Moment(args)
                self.moment.init_moment(args, encoder, train_data_for_initial, is_memory=False)
                ## Todo
                # Self-purified train_data_for_initial
                """
                if len(memorized_samples) > 0: # not first task
                    idex2relation=dict()
                    train_data_seen = []
                    train_data_unseen = []
                    past_relations = [t for t in history_relation if t not in current_relations]
                    print("past relations:")
                    print(past_relations)
                    past_rel_ids = [self.rel2id[r] for r in past_relations]
                    for relation in current_relations:
                        for sample in training_data[relation]:
                            if sample['ori_relation'] in past_rel_ids:
                                train_data_seen.append(sample)
                            else:
                                train_data_unseen.append(sample)
                    old_protos = []
                    for idx, relation in enumerate(past_relations):
                        old_proto,_ = self.get_proto(args,encoder,memorized_samples[relation])
                        idex2relation[idx] = relation
                        old_protos.append(old_proto)
                    old_protos = torch.cat(old_protos, dim=0) # current prototypes
                    print(len(train_data_seen),len(train_data_unseen),len(train_data_for_initial))
                    if len(train_data_seen)>0:
                        print(train_data_seen[0])
                        self.compute_proto_dist(args,encoder,steps,old_protos,train_data_seen,idex2relation,seen=True)
                    if len(train_data_unseen)>0:
                        print(train_data_unseen[0])
                        self.compute_proto_dist(args,encoder,steps,old_protos,train_data_unseen,idex2relation,seen=False)
                """
                """
                seen_noisy_labels / prototypes

                unseen_noisy_labels / prototypes
                current_train_data / prototypes
                """
                self.train_simple_model(args, encoder, train_data_for_initial, args.step1_epochs, steps+1, self.tensor_writer)
                
                # repaly
                if len(memorized_samples)>0: # not the first class
                    # select current task sample
                    for relation in current_relations:
                        memorized_samples[relation], _, _ = self.select_data(args, encoder, training_data[relation]) # 选择当前任务最有代表性的样本，存入memory
                        ## print("memory size for relation {} is".format(relation))
                        # print(len(memorized_samples[relation]))

                    train_data_for_memory = []
                    for relation in history_relation:
                        train_data_for_memory += memorized_samples[relation] # train_data_for_memory 包含了之前的+当前的各关系的memorized_samples
                    
                    print("length of memory data is :", len(train_data_for_memory))
                    
                    self.moment.init_moment(args, encoder, train_data_for_memory, is_memory=True)# 用之前的+当前的各关系的memorized_samples来更新memory（存一下选出来的样本对应的64维向量和标签）
                    self.train_mem_model(args, encoder, train_data_for_memory, proto4repaly, args.step2_epochs, seen_relations)

                # 测试当前任务的accuracy & 历史任务的average accuracy
                feat_mem = []
                proto_mem = []

                for relation in current_relations:
                    memorized_samples[relation], feat, temp_proto = self.select_data(args, encoder, training_data[relation])
                    feat_mem.append(feat)
                    proto_mem.append(temp_proto)

                feat_mem = torch.cat(feat_mem, dim=0)
                temp_proto = torch.stack(proto_mem, dim=0)
                #memory evaluate
                flip_cnt = 0
                memorized_samples_cnt = 0
                for relation in seen_relations:
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
                        
                        protos, featrues = self.get_proto(args, encoder, memorized_samples[relation])
                        protos4eval.append(protos)
                        featrues4eval.append(featrues)
                
                if protos4eval: # not first class
                    
                    protos4eval = torch.cat(protos4eval, dim=0).detach()
                    protos4eval = torch.cat([protos4eval, temp_proto.to(args.gpu)], dim=0)

                else: # first class
                    protos4eval = temp_proto.to(args.gpu)
                proto4repaly = protos4eval.clone()

                test_data_2 = []
                for relation in seen_relations:
                    test_data_2 += historic_test_data[relation]

                acc_list,total_acc = self.evaluate_strict_model(args, encoder, test_data_2, protos4eval, featrues4eval,seen_relations)
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
                        forget_rate = max(his_acc) - cur_acc
                        forget_list.append(forget_rate)
                test_forget.append(forget_list)
                print(test_cur)
                print(test_total)
                print(test_acc)
                print(f'forget rate:')
                print(test_forget)
                del self.moment
