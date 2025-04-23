import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
from torch.nn import functional as F


class Conv_TA(torch.nn.Module):
    def __init__(self, dataset,params):
        super(Conv_TA, self).__init__()
        self.p                  = params

        self.dataset = dataset
        self.ent_embs      = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.ent_embed      = nn.Embedding(dataset.numEnt(), params.s_emb_dim+params.t_emb_dim).cuda()
        self.rel_embs      = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)
        nn.init.xavier_normal_(self.ent_embed.weight)
        self.t_rel_embs = nn.Embedding(dataset.numRel(), params.t_emb_dim).cuda()
        self.time_embs_dim = self.p.time_embs_dim
        self.out_dim = int(self.p.num_filt * self.p.perm)
        self.m_ratio=self.p.m_ratio
        self.d_ratio=self.p.d_ratio
        self.y_embs = nn.Embedding(3000, self.time_embs_dim-int(self.d_ratio*self.time_embs_dim)-int(self.m_ratio*self.time_embs_dim)).cuda()
        self.m_embs = nn.Embedding(13, int(self.m_ratio*self.time_embs_dim)).cuda()
        self.d_embs = nn.Embedding(32, int(self.d_ratio*self.time_embs_dim)).cuda()
        self.t_linner = torch.nn.Linear(self.time_embs_dim, self.out_dim)
        self.t_drop	= torch.nn.Dropout(self.p.t_drop)

        nn.init.xavier_uniform_(self.y_embs.weight)
        nn.init.xavier_uniform_(self.m_embs.weight)
        nn.init.xavier_uniform_(self.d_embs.weight)
        self.sigm = torch.nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.inp_drop		= torch.nn.Dropout(params.input_drop)
        self.hidden_drop	= torch.nn.Dropout(params.hidden_drop)
        self.feature_map_drop	= torch.nn.Dropout2d(params.feat_drop)
        self.bn0		= torch.nn.BatchNorm2d(self.p.perm)
        self.flat_sz_h 		= self.p.k_h
        self.flat_sz_w 		= 2*self.p.k_w
        self.padding 		= 0
        self.bn1 		= torch.nn.BatchNorm2d(self.p.num_filt*self.p.perm)
        self.flat_sz 		= self.flat_sz_h * self.flat_sz_w * self.p.num_filt*self.p.perm
        self.bn2		= torch.nn.BatchNorm1d(self.p.emb_dim)
        self.fc 		= torch.nn.Linear(self.flat_sz, self.p.emb_dim)
        self.chequer_perm	= self.get_chequer_perm()
        self.register_parameter('bias', Parameter(torch.zeros(dataset.numEnt())))
        self.register_parameter('conv_filt', Parameter(torch.zeros(self.p.num_filt, 1, self.p.ker_sz,  self.p.ker_sz))); xavier_normal_(self.conv_filt)


    def getEmbeddings(self, heads, rels):
        r = self.rel_embs(rels)
        h = self.ent_embed(heads)
        return h,r

    def get_time_emb(self,years,months,days):
        y = self.y_embs(years.long()).squeeze(1)
        m = self.m_embs(months.long()).squeeze(1)
        d = self.d_embs(days.long()).squeeze(1)
        time_emb = torch.cat([y, m, d], dim=1)
        return time_emb

    def get_chequer_perm(self):
        ent_perm  = np.int32([np.random.permutation(self.p.emb_dim) for _ in range(self.p.perm)])
        rel_perm  = np.int32([np.random.permutation(self.p.emb_dim) for _ in range(self.p.perm)])

        comb_idx = []
        for k in range(self.p.perm):
            temp = []
            ent_idx, rel_idx = 0, 0

            for i in range(self.p.k_h):
                for j in range(self.p.k_w):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
                            temp.append(rel_perm[k, rel_idx]+self.p.emb_dim); rel_idx += 1;
                        else:
                            temp.append(rel_perm[k, rel_idx]+self.p.emb_dim); rel_idx += 1;
                            temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx]+self.p.emb_dim); rel_idx += 1;
                            temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
                        else:
                            temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
                            temp.append(rel_perm[k, rel_idx]+self.p.emb_dim); rel_idx += 1;

            comb_idx.append(temp)

        chequer_perm = torch.LongTensor(np.int32(comb_idx)).cuda()
        return chequer_perm
    def circular_padding_chw(self, batch, padding):
        upper_pad	= batch[..., -padding:, :]
        lower_pad	= batch[..., :padding, :]
        temp		= torch.cat([upper_pad, batch, lower_pad], dim=2)

        left_pad	= temp[..., -padding:]
        right_pad	= temp[..., :padding]
        padded		= torch.cat([left_pad, temp, right_pad], dim=3)
        return padded

    def forward(self, heads, rels, tails, years, months, days):
        h_embs, r_embs = self.getEmbeddings(heads, rels)
        sub_emb		= h_embs
        rel_emb		= r_embs
        comb_emb	= torch.cat([sub_emb, rel_emb], dim=1)
        chequer_perm	= comb_emb[:, self.chequer_perm]
        stack_inp	= chequer_perm.reshape((-1, self.p.perm, 2*self.p.k_w, self.p.k_h))
        stack_inp	= self.bn0(stack_inp)
        x		= self.inp_drop(stack_inp)
        x		= self.circular_padding_chw(x, self.p.ker_sz//2)
        x		= F.conv2d(x, self.conv_filt.repeat(self.p.perm, 1, 1, 1), padding=self.padding, groups=self.p.perm)
        x		= self.bn1(x)
        x		= F.relu(x)
        t_embed = self.get_time_emb(years,months,days)
        t_weights = self.t_linner(t_embed)
        t_weights = self.t_drop(t_weights)
        t_weights = torch.sigmoid(t_weights.squeeze(1))
        t_weights = t_weights.unsqueeze(-1).unsqueeze(-1)
        x = x * t_weights
        x		= self.feature_map_drop(x)
        x		= x.view(-1, self.flat_sz)
        x		= self.fc(x)
        x		= self.hidden_drop(x)
        x		= self.bn2(x)
        x		= F.relu(x)
        x = torch.mm(x, self.ent_embed.weight.transpose(1,0))
        x += self.bias.expand_as(x)
        pred	= torch.sigmoid(x)

        return pred