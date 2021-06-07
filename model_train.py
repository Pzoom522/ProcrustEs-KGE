#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import numpy as np

class ProcrustEs(nn.Module):
    def __init__(self, rel_ent_dict, nentity, nrelation, td, sd, is_cuda, save_path, eps):
        super(ProcrustEs, self).__init__()
        self.rel_ent_dict = rel_ent_dict
        self.nentity = nentity
        self.nrelation = nrelation
        self.td = td
        self.sd = sd
        self.sub_num = int(td / sd)
        self.save_path = save_path

        self.is_cuda = is_cuda

        self.map_dict = {}

        if self.is_cuda:
            self.safety_norm = eps * torch.ones(self.sub_num, self.nentity, 1).cuda()
            self.safety_div = eps * torch.ones(self.sub_num, 1, 1).cuda()
            self.safety_svd = eps * torch.eye(self.sd).cuda()
        else:
            self.safety_norm = eps * torch.ones(self.sub_num, self.nentity, 1)
            self.safety_div = eps * torch.ones(self.sub_num, 1, 1)
            self.safety_svd = eps * torch.eye(self.sd)

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.td), requires_grad=True)
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-1.0,
            b=1.0
        )

    def normalise(self):
        # mean centering
        mean_ent_emb = self.entity_embedding - self.entity_embedding.mean(0)
        mean_ent_emb = torch.stack(mean_ent_emb.chunk(self.sub_num, dim=1)) 
        # sub_num * nentity * sd

        # unit length
        mean_ent_emb = mean_ent_emb / (mean_ent_emb.norm(dim=2).view(self.sub_num, self.nentity, 1) + self.safety_norm)
        mean_ent_emb = torch.cat(mean_ent_emb.split(1), 2).view(self.nentity, self.td)
        self.entity_embedding.data = mean_ent_emb

    def forward(self, save=False):
        if self.is_cuda:
            loss_list = torch.zeros(self.sub_num).cuda()
            place_holder = torch.zeros(self.sub_num).cuda()
        else:
            loss_list = torch.zeros(self.sub_num)
            place_holder = torch.zeros(self.sub_num)

        for rel in self.rel_ent_dict:
            head_mask = self.rel_ent_dict[rel][:, 0]
            tail_mask = self.rel_ent_dict[rel][:, 1]
            head_emb = torch.stack(self.entity_embedding[head_mask].chunk(self.sub_num, dim=1))
            tail_emb = torch.stack(self.entity_embedding[tail_mask].chunk(self.sub_num, dim=1))

            A = tail_emb.transpose(1, 2).bmm(head_emb)
            U, s, V = torch.svd(A + self.safety_svd)
            T = V.bmm(U.transpose(1, 2))
            norm = (head_emb.bmm(T) - tail_emb).norm(dim=(1, 2))


            if save:
                self.map_dict[rel] = T

            loss_list = loss_list + norm
           
        if save:
            entity_embedding = self.entity_embedding.detach().cpu().numpy()
            np.save(os.path.join(self.save_path, 'entity_embedding.npy'), entity_embedding)

            relation_embedding = torch.zeros(self.nrelation, self.td * self.sd)
            for rel in self.map_dict:
                relation_embedding[rel] = self.map_dict[rel].view(-1)
            np.save(os.path.join(self.save_path, 'relation_embedding.npy'), relation_embedding.detach().cpu().numpy())
        
        loss = torch.nn.L1Loss(reduction='sum')   

        return loss(loss_list / len(self.rel_ent_dict), place_holder)
            