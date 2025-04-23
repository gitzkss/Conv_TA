# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
class Params:

    def __init__(self,
                 ne=1000,
                 bsize=128,
                 lr=0.001,
                 reg_lambda=0.0,
                 embedding_shape1=20,
                 hidden_drop=0.3,
                 input_drop=0.2,
                 feat_drop=0.2,
                 hidden_size=9728,
                 label_smoothing=0.1,
                 emb_dim=200,
                 neg_ratio=20,
                 dropout=0.4,
                 save_each=50,
                 se_prop=0.9,
                 ker_sz=9,
                 k_w=10,
                 k_h=20,
                 perm=1,
                 num_filt=96,
                 alp=0.5,
                 t_ratio = 0.7,
                 m_ratio = 0.3,
                 d_ratio = 0.4,
                 num_heads = 16,
                 time_embs_dim = 100,
                 t_drop = 0.2,
                 model="Conv_TA"
                 ):
        self.ne = ne
        self.bsize = bsize
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.s_emb_dim = int(se_prop*emb_dim)
        self.t_emb_dim = emb_dim - int(se_prop*emb_dim)
        self.save_each = save_each
        self.neg_ratio = neg_ratio
        self.dropout = dropout
        self.se_prop = se_prop
        self.embedding_shape1 = embedding_shape1
        self.hidden_drop = hidden_drop
        self.input_drop = input_drop
        self.feat_drop = feat_drop
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.label_smoothing=label_smoothing
        self.ker_sz= ker_sz
        self.k_w = k_w
        self.k_h = k_h
        self.perm = perm
        self.num_filt=num_filt
        self.dim_ff = self.t_emb_dim * 2
        self.alp = alp
        self.n_head = 4
        self.d_k = self.t_emb_dim // self.n_head
        self.d_v = self.t_emb_dim // self.n_head
        self.t_ratio = t_ratio
        self.m_ratio = m_ratio
        self.d_ratio = d_ratio
        self.num_heads = num_heads
        self.time_embs_dim = time_embs_dim
        self.t_drop = t_drop
        self.model = model
    def str_(self):
        return str(self.ne) + "_" + str(self.bsize) + "_" + str(self.lr) + "_" + str(self.reg_lambda) + "_" + str(self.s_emb_dim) + "_" + str(self.neg_ratio) + "_" + str(self.dropout) + "_" + str(self.t_emb_dim) + "_" + str(self.save_each) + "_" + str(self.se_prop) 