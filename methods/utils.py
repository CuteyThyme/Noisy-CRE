from turtle import xcor
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm, trange
import random
import math

from dataloaders.data_loader import get_data_loader

class Moment:
    def __init__(self, args) -> None:
        
        self.labels = None
        self.mem_labels = None
        self.memlen = 0
        self.sample_k = 500
        self.temperature= args.temp
        self.device = args.gpu
        self.margin = args.margin

    def get_mem_proto(self):
        c = self._compute_centroids_ind()
        return c
    
    def _compute_centroids_ind(self):
        cinds = []
        for x in self.mem_labels:
            if x.item() not in cinds:
                cinds.append(x.item())

        num = len(cinds)
        feats = self.mem_features
        centroids = torch.zeros((num, feats.size(1)), dtype=torch.float32, device=feats.device)
        for i, c in enumerate(cinds):
            ind = np.where(self.mem_labels.cpu().numpy() == c)[0] # 定位index
            centroids[i, :] = F.normalize(feats[ind, :].mean(dim=0), p=2, dim=0) #找到对应的特征，保存
        return centroids

    def update(self, ind, feature, init=False):
        self.features[ind] = feature
        
    def update_mem(self, ind, feature, hidden=None):
        self.mem_features[ind] = feature
        if hidden is not None:
            self.hidden_features[ind] = hidden
            
    @torch.no_grad()
    def init_moment(self, args, encoder, datasets, noisy_datasets=None, noisy_embeddings=None, is_memory=False):
        encoder.eval()
        if noisy_datasets is not None:
            datalen = len(datasets) + len(noisy_datasets)
        else:
            datalen = len(datasets)
        if not is_memory:
            self.features = torch.zeros(datalen, args.feat_dim).to(self.device)
            data_loader = get_data_loader(args, datasets)
            td = tqdm(data_loader)
            lbs = []
            # update with clean data
            for step, batch_data in enumerate(td):
                labels, tokens, flag, ind = batch_data
                tokens = torch.stack([x.to(self.device) for x in tokens], dim=0)
                if encoder.hidden is False:
                    _, reps = encoder.bert_forward(tokens)
                else:
                    _, reps, _ = encoder.bert_forward(tokens)
                self.update(ind, reps.detach())
                lbs.append(labels)
            # update with noisy data
            if noisy_datasets is not None:
                noisy_data_loader = get_data_loader(args, noisy_datasets, shuffle=False)
                noisy_td = tqdm(noisy_data_loader) 
                for noisy_embedding, batch_data in zip(noisy_embeddings, noisy_td):
                    labels, tokens, flag, ind = batch_data
                    tokens = torch.stack([x.to(self.device) for x in tokens], dim=0)
                    noisy_embedding = noisy_embedding.to(self.device)
                    if encoder.hidden is False:
                        _, reps = encoder.bert_forward(tokens,flag=True,embedding=noisy_embedding)
                    else:
                        _, reps, _ = encoder.bert_forward(tokens)
                    self.update(ind, reps.detach())
                    lbs.append(labels)
            lbs = torch.cat(lbs)
            self.labels = lbs.to(self.device)
        else:
            self.memlen = datalen
            self.mem_features = torch.zeros(datalen, args.feat_dim).to(self.device)
            self.hidden_features = torch.zeros(datalen, args.encoder_output_size).to(self.device)
            lbs = []
            data_loader = get_data_loader(args, datasets)
            td = tqdm(data_loader)
            for step, batch_data in enumerate(td):
                labels, tokens, flag, ind = batch_data
                tokens = torch.stack([x.to(self.device) for x in tokens], dim=0)
                if encoder.hidden is True:
                    hidden, reps, _ = encoder.bert_forward(tokens)
                else:
                    hidden, reps = encoder.bert_forward(tokens)
                self.update_mem(ind, reps.detach(), hidden.detach())
                lbs.append(labels)
            lbs = torch.cat(lbs)
            self.mem_labels = lbs.to(self.device)

    def loss(self, x, labels, is_mem=False, mapping=None):
        if is_mem:
            ct_x = self.mem_features
            ct_y = self.mem_labels
        else:
            if self.sample_k is not None:
            # sample some instances
                idx = list(range(len(self.features)))
                if len(idx) > self.sample_k:
                    sample_id = random.sample(idx, self.sample_k)
                else:
                    sample_id = idx
                ct_x = self.features[sample_id]
                ct_y = self.labels[sample_id]
            else:
                ct_x = self.features
                ct_y = self.labels
        
        dot_product_tempered = torch.mm(x, ct_x.T) / self.temperature  # n * m
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0].detach()) + 1e-5
        )
        
        mask_combined = (labels.unsqueeze(1).repeat(1, ct_y.shape[0]) == ct_y).to(self.device) # n*m
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered, dim=1, keepdim=True)))
        
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
        
        return supervised_contrastive_loss

    def amc_loss(self, x, labels, is_mem=False, mapping=None):
        if is_mem:
            ct_x = self.mem_features
            ct_y = self.mem_labels
        else:
            if self.sample_k is not None:
            # sample some instances
                idx = list(range(len(self.features)))
                if len(idx) > self.sample_k:
                    sample_id = random.sample(idx, self.sample_k)
                else:
                    sample_id = idx
                ct_x = self.features[sample_id]
                ct_y = self.labels[sample_id]
            else:
                ct_x = self.features
                ct_y = self.labels
                
        dot_product = torch.mm(x, ct_x.T)  # [batch of x * batch of ct_x]
        #cos_dot_product = dot_product / (torch.sqrt(torch.sum(torch.pow(x, 2))) * torch.sqrt(torch.sum(torch.pow(ct_x, 2)))) # [batch of x * batch of ct_x]
        cos_dot_product = dot_product # [batch of x * batch of ct_x]
        alpha_dot_product = torch.acos(cos_dot_product) # [batch of x * batch of ct_x]
        margin_dot_product_tempered = torch.cos(alpha_dot_product + math.radians(self.margin)) / self.temperature # [batch of x * batch of ct_x]
        dot_product_tempered = torch.cos(alpha_dot_product) / self.temperature # [batch of x * batch of ct_x]
        exp_margin_dot_tempered = (
            torch.exp(margin_dot_product_tempered - torch.max(margin_dot_product_tempered, dim=1, keepdim=True)[0].detach()) + 1e-5 # [batch of x * batch of ct_x]
        ) # [batch of x * batch of ct_x]
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0].detach()) + 1e-5 # [batch of x * batch of ct_x]
        ) # [batch of x * batch of ct_x]
        mask_combined = (labels.unsqueeze(1).repeat(1, ct_y.shape[0]) == ct_y) # [batch of x * batch of ct_x], 指示一下标签是否一样
        reversed_mask_combined = ~mask_combined
        cardinality_per_samples = torch.sum(mask_combined, dim=1) # 标签一样的个数 [batch of x] 
        log_prob = -torch.log(exp_margin_dot_tempered / (torch.sum(exp_margin_dot_tempered * mask_combined + exp_dot_tempered * reversed_mask_combined, dim=1, keepdim=True))) # [batch of x * batch of ct_x]
        arc_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples # batch of x
        arc_contrastive_loss = torch.mean(arc_contrastive_loss_per_sample) # a tensor
        
        return arc_contrastive_loss
    
    def amc_loss_v2(self, x, labels, is_mem=False, mapping=None):
        dot_product = torch.mm(x, x.T)  # [batch of x * batch of ct_x]
        #cos_dot_product = dot_product / (torch.sqrt(torch.sum(torch.pow(x, 2))) * torch.sqrt(torch.sum(torch.pow(ct_x, 2)))) # [batch of x * batch of ct_x]
        cos_dot_product = dot_product # [batch of x * batch of ct_x]
        epsilon=1e-7 # to avoid nan when approaches 1 or -1
        alpha_dot_product = torch.acos(torch.clamp(cos_dot_product, -1 + epsilon, 1 - epsilon)) # [batch of x * batch of ct_x]
        #print("alpha_dot_product")
        #print(alpha_dot_product)
        margin_dot_product_tempered = torch.cos(alpha_dot_product + self.margin) / self.temperature # [batch of x * batch of ct_x]
        dot_product_tempered = torch.cos(alpha_dot_product) / self.temperature # [batch of x * batch of ct_x]
        exp_margin_dot_tempered = (
            torch.exp(margin_dot_product_tempered - torch.max(margin_dot_product_tempered, dim=1, keepdim=True)[0].detach()) + 1e-5 # [batch of x * batch of ct_x]
        ) # [batch of x * batch of ct_x]
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0].detach()) + 1e-5 # [batch of x * batch of ct_x]
        ) # [batch of x * batch of ct_x]
        mask_combined = (labels.unsqueeze(1).repeat(1, labels.shape[0]) == labels) # [batch of x * batch of ct_x], 指示一下标签是否一样
        reversed_mask_combined = ~mask_combined
        cardinality_per_samples = torch.sum(mask_combined, dim=1) # 标签一样的个数 [batch of x] 
        log_prob = -torch.log(exp_margin_dot_tempered / (torch.sum(exp_margin_dot_tempered * mask_combined + exp_dot_tempered * reversed_mask_combined, dim=1, keepdim=True))) # [batch of x * batch of ct_x]
        arc_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples # batch of x
        arc_contrastive_loss = torch.mean(arc_contrastive_loss_per_sample) # a tensor
        return arc_contrastive_loss

    def amc_loss_v3(self, x, labels, aug_x, flag=False, is_mem=False, mapping=None):
        if is_mem:
            ct_x = self.mem_features
            ct_y = self.mem_labels
        else:
            if self.sample_k is not None:
            # sample some instances
                idx = list(range(len(self.features)))
                if len(idx) > self.sample_k:
                    sample_id = random.sample(idx, self.sample_k)
                else:
                    sample_id = idx
                ct_x = self.features[sample_id]
                ct_y = self.labels[sample_id]
            else:
                ct_x = self.features
                ct_y = self.labels
        all_x = torch.cat((ct_x,aug_x),dim=0)
        all_y = torch.cat((ct_y,labels),dim=0)
        dot_product = torch.mm(x, all_x.T)  # [batch of x * batch of ct_x]
        #cos_dot_product = dot_product / (torch.sqrt(torch.sum(torch.pow(x, 2))) * torch.sqrt(torch.sum(torch.pow(ct_x, 2)))) # [batch of x * batch of ct_x]
        cos_dot_product = dot_product # [batch of x * batch of ct_x]
        epsilon=1e-7 # to avoid nan when approaches 1 or -1
        alpha_dot_product = torch.acos(torch.clamp(cos_dot_product, -1 + epsilon, 1 - epsilon)) # [batch of x * batch of ct_x]
        #print("alpha_dot_product")
        #print(alpha_dot_product)
        if not flag: # default no margin
            margin_dot_product_tempered = torch.cos(alpha_dot_product) / self.temperature # [batch of x * batch of ct_x]
        else:
            margin_dot_product_tempered = torch.cos(alpha_dot_product + self.margin) / self.temperature
        dot_product_tempered = torch.cos(alpha_dot_product) / self.temperature # [batch of x * batch of ct_x]
        exp_margin_dot_tempered = (
            torch.exp(margin_dot_product_tempered - torch.max(margin_dot_product_tempered, dim=1, keepdim=True)[0].detach()) + 1e-5 # [batch of x * batch of ct_x]
        ) # [batch of x * batch of ct_x]
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0].detach()) + 1e-5 # [batch of x * batch of ct_x]
        ) # [batch of x * batch of ct_x]
        mask_combined = (labels.unsqueeze(1).repeat(1, all_y.shape[0]) == all_y) # [batch of x * batch of ct_x], 指示一下标签是否一样
        reversed_mask_combined = ~mask_combined
        cardinality_per_samples = torch.sum(mask_combined, dim=1) # 标签一样的个数 [batch of x] 
        log_prob = -torch.log(exp_margin_dot_tempered / (torch.sum(exp_margin_dot_tempered * mask_combined + exp_dot_tempered * reversed_mask_combined, dim=1, keepdim=True))) # [batch of x * batch of ct_x]
        arc_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples # batch of x
        arc_contrastive_loss = torch.mean(arc_contrastive_loss_per_sample) # a tensor
        return arc_contrastive_loss

    def amc_loss_adversarial(self, x, labels, aug_x, noisy_reps=None, is_neg=None, noisy_labels=None, margin_dict=None, flag=False, is_mem=False, mapping=None):
        if is_mem:
            ct_x = self.mem_features
            ct_y = self.mem_labels
        else:
            if self.sample_k is not None:
            # sample some instances
                idx = list(range(len(self.features)))
                if len(idx) > self.sample_k: # clean_data
                    sample_id = random.sample(idx, self.sample_k)
                else:
                    sample_id = idx
                ct_x = self.features[sample_id]
                ct_y = self.labels[sample_id]
            else:
                ct_x = self.features
                ct_y = self.labels
        if noisy_reps is not None:
            idx = list(range(len(noisy_reps)))
            if len(idx) > 15: # sample 200 noisy samples ～ 15 batches
                sample_id = random.sample(idx, 15)
            else:
                sample_id = idx
            adv_x = [noisy_reps[i] for i in sample_id]
            neg_flag = [is_neg[i] for i in sample_id]
            adv_y = [noisy_labels[i] for i in sample_id]
            yy = list()
            for y,flag_ in zip(adv_y,neg_flag):
                yy.append([i_y if i_flag.item() is False else -i_y for (i_y,i_flag) in zip(y,flag_)])
            print("yy:")
            print(yy[0])
            adv_x = torch.cat(adv_x,dim=0)
            adv_y = torch.cat([x for x in [torch.stack(y,0) for y in yy]], dim=0).flatten()
            print("adv ct shape:")
            print(adv_x.shape,adv_y.shape)
            print("positive samples in {} adv samples are {}".format(adv_y.shape[0],torch.sum(adv_y>0).item())) 

        if noisy_reps is not None:
            all_x = torch.cat((ct_x,aug_x,adv_x),dim=0)
            all_y = torch.cat((ct_y,labels,adv_y),dim=0)
        else:
            print("without noisy embeddings...")
            all_x = torch.cat((ct_x,aug_x),dim=0)
            all_y = torch.cat((ct_y,labels),dim=0)
        print("all_x,y:")
        print(all_x.shape,all_y.shape)
        dot_product = torch.mm(x, all_x.T)  # [batch of x * batch of ct_x]
        #cos_dot_product = dot_product / (torch.sqrt(torch.sum(torch.pow(x, 2))) * torch.sqrt(torch.sum(torch.pow(ct_x, 2)))) # [batch of x * batch of ct_x]
        cos_dot_product = dot_product # [batch of x * batch of ct_x]
        epsilon=1e-7 # to avoid nan when approaches 1 or -1
        alpha_dot_product = torch.acos(torch.clamp(cos_dot_product, -1 + epsilon, 1 - epsilon)) # [batch of x * batch of ct_x]
        #print("alpha_dot_product")
        #print(alpha_dot_product)
        if not flag or margin_dict is None: # default no margin
            margin_dot_product_tempered = torch.cos(alpha_dot_product) / self.temperature # [batch of x * batch of ct_x]
        else:
            margins = torch.stack([margin_dict[label.item()] for label in labels])
            margins = margins.unsqueeze(1).repeat(1,all_x.shape[0]) # [batch of x * batch of ct_x]
            print("shape of margins")
            print(margins.shape)
            margin_dot_product_tempered = torch.cos(alpha_dot_product + margins) / self.temperature
            #margin_dot_product_tempered = torch.cos(alpha_dot_product + 0) / self.temperature
        dot_product_tempered = torch.cos(alpha_dot_product) / self.temperature # [batch of x * batch of ct_x]
        exp_margin_dot_tempered = (
            torch.exp(margin_dot_product_tempered - torch.max(margin_dot_product_tempered, dim=1, keepdim=True)[0].detach()) + 1e-5 # [batch of x * batch of ct_x]
        ) # [batch of x * batch of ct_x]
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0].detach()) + 1e-5 # [batch of x * batch of ct_x]
        ) # [batch of x * batch of ct_x]
        mask_combined = (labels.unsqueeze(1).repeat(1, all_y.shape[0]) == all_y) # [batch of x * batch of ct_x], 指示一下标签是否一样
        reversed_mask_combined = ~mask_combined 
        cardinality_per_samples = torch.sum(mask_combined, dim=1) # 标签一样的个数 [batch of x] 
        ## Remedy
        log_prob = -torch.log(exp_margin_dot_tempered / (torch.sum(exp_margin_dot_tempered * mask_combined + exp_dot_tempered * reversed_mask_combined, dim=1, keepdim=True))) # [batch of x * batch of ct_x]
        arc_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples # batch of x
        arc_contrastive_loss = torch.mean(arc_contrastive_loss_per_sample) # a tensor
        return arc_contrastive_loss

def dot_dist(x1, x2):
    return torch.matmul(x1, x2.t())

def osdist(x, c):
    pairwise_distances_squared = torch.sum(x ** 2, dim=1, keepdim=True) + \
                                 torch.sum(c.t() ** 2, dim=0, keepdim=True) - \
                                 2.0 * torch.matmul(x, c.t())
    error_mask = pairwise_distances_squared <= 0.0
    pairwise_distances = pairwise_distances_squared.clamp(min=1e-16)#.sqrt()
    pairwise_distances = torch.mul(pairwise_distances, ~error_mask)

    return pairwise_distances
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output