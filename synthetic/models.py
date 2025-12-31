import os
import torch
import random
import argparse

'''
####################################
    Sequentially estimated model 
####################################

Two ways of enforcing conditional independences are proposed for the following two models, respectively:

Op1. SeqEstARModel_PreZT:
    z_2^{t+1} \indep x^t | z_2^{t}, z_1^{t}

Op2. SeqEstARModel_CurrentPreZT
    z_2^{t+1} \indep x^t | z_2^{t}, z_1^{t+1}

Note that Op2 is more strictly held in the structure that our proposed only, 
while Op1 allows there are temporal causal effect acorss layers as well.

Therefore, in Op2, H(z_2^{t+1}|z_2^{t}, z_1^{t+1}) is a direct representation of the entropy of e_2^{t+1}, 
while in Op1, H(z_2^{t+1}|z_2^{t}, z_1^{t}) includes the entropy from e_1^{t+1} as well with some transformation from the instantaneous causal effect from z_1^{t+1} to z_2^{t+1}.

But technically, both CI could be used to identify zt layer by layer. 
'''

class SeqEstARModel_PreZT(torch.nn.Module): 
    def __init__(self, nlags, xt_dim, est_zt_dim, hid_dim, n_pre_zt, dropout=0.5, poc_flag=False):
        super(SeqEstARModel_PreZT, self).__init__()

        self.nlags = nlags
        self.xt_dim = xt_dim
        self.est_zt_dim = est_zt_dim # the dim of the zt to estimate; assuming the hidden dim for each layer is the same
        self.hid_dim = hid_dim
        self.n_pre_zt = n_pre_zt 
        self.dropout = dropout
        self.poc_flag = poc_flag # this model can also be used as proof oc concept only, and can take gt_zt as input during training

        # define the mapping from xt to est_zt
        self.h = torch.nn.Sequential(
                torch.nn.Linear(self.xt_dim, self.hid_dim),
                torch.nn.BatchNorm1d(self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hid_dim, self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hid_dim, self.est_zt_dim)
            )
        
        # define the pseudo inverse of h_i
        self.h_inv = torch.nn.Sequential(
                    torch.nn.Linear(self.est_zt_dim * (self.n_pre_zt + 1), self.hid_dim),
                    torch.nn.BatchNorm1d(self.hid_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(self.dropout),
                    torch.nn.Linear(self.hid_dim, self.hid_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hid_dim, self.xt_dim)
            )
        
        # define the AR model on xt
        self.m = torch.nn.Sequential(
                torch.nn.Linear(self.xt_dim * self.nlags, self.hid_dim),
                torch.nn.BatchNorm1d(self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hid_dim, self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hid_dim, self.xt_dim)
            )

        # define the AR model on zt 
        self.f = torch.nn.Sequential(
                torch.nn.Linear(self.est_zt_dim * (self.n_pre_zt+1) * self.nlags, self.hid_dim),
                torch.nn.BatchNorm1d(self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hid_dim, self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hid_dim, self.est_zt_dim)
            )

    def forward(self, xt, pre_zt_list=None, zt=None):
        '''
        1. xt is a torch.tensor of shape: (seq_length, xt_dim, batch_size)
        2. pre_zt_list (optional; will not be given when estimate z1) is a list of previously estimated zt [(seq_length, xt_dim, batch_size),...], as each element is a tranformed xt on the xt_dim dimension 
        3. zt (optional) is a torch.tensor of shape: (seq_length, est_zt_dim, batch_size)6
        '''
        if pre_zt_list is None:
            pre_zt_list = []
        
        if zt is not None:
            self.poc_flag = True

        length, xt_dim, batch_size = xt.shape
        assert xt_dim == self.xt_dim
        assert xt_dim > self.est_zt_dim

        n_pre_zt = len(pre_zt_list)
        if n_pre_zt > 0:
            assert n_pre_zt == self.n_pre_zt
            for pre_zt in pre_zt_list:
                _length, _zt_dim, _batch_size = pre_zt.shape
                assert _length == length
                assert _batch_size == batch_size
                assert _zt_dim == self.est_zt_dim # assuming the dim of all zts are the same
        
        ### CI constraint: estimate z_{t+1} (zt_p1) from h, m ###
        # part 1: h(xt)
        xt_perm_flatten = xt.permute(0, 2, 1).reshape(-1, self.xt_dim)
        if not self.poc_flag:
            zt = self.h(xt_perm_flatten).reshape(length, batch_size, self.est_zt_dim).permute(0, 2, 1)

        # part 2: h * m (xt)
        m_xt_p1 = []
        for t in range(self.nlags, length):
            xt_history = xt[t-self.nlags:t, :, :]
            xt_history_perm_flatten = xt_history.permute(2, 1, 0).reshape(batch_size, self.xt_dim * self.nlags)
            m_xt_history = self.m(xt_history_perm_flatten).reshape(batch_size, self.xt_dim).permute(1, 0)
            m_xt_p1.append(m_xt_history)
        m_xt_p1 = torch.stack(m_xt_p1, dim=0) # length = length - nlags
        h_m_xt_p1_perm_flatten = self.h(m_xt_p1.permute(0, 2, 1).reshape(-1, self.xt_dim))
        h_m_xt_p1 = h_m_xt_p1_perm_flatten.reshape(length - self.nlags, batch_size, self.est_zt_dim).permute(0, 2, 1)

        ### CI constraint: estimate z_{t+1} (zt_p1) from f, h ###
        # part 1: reused
        # part 2: f * h (xt) = f(zt)
        f_zt_p1 = []
        for t in range(self.nlags, length):
            zt_history = zt[t-self.nlags:t, :, :]
            zt_history_perm_flatten = zt_history.permute(2, 0, 1).reshape(-1, self.est_zt_dim * self.nlags)
            if self.n_pre_zt > 0:
                pre_zt_history_perm_flatten = []
                for pre_zt in pre_zt_list:
                    pre_zt_history_perm_flatten.append(pre_zt[t-self.nlags:t, :, :].permute(2, 0, 1).reshape(-1, self.est_zt_dim * self.nlags))
                pre_zt_history_perm_flatten = torch.cat(pre_zt_history_perm_flatten, dim=-1)
                zt_history_perm_flatten = torch.cat([zt_history_perm_flatten, pre_zt_history_perm_flatten], dim=-1)
            f_zt_history = self.f(zt_history_perm_flatten).permute(1, 0)            
            f_zt_p1.append(f_zt_history)
        f_zt_p1 = torch.stack(f_zt_p1, dim=0) # length = length - nlags

        # MI constraint: reconstruct xt from zt_p1 and pre_zt_p1
        zt_p1 = zt[self.nlags:, :, :]
        zt_p1_perm_flatten = zt_p1.permute(0, 2, 1).reshape(-1, self.est_zt_dim)
        if self.n_pre_zt > 0:
            for pre_zt in pre_zt_list:
                _pre_zt_p1 = pre_zt[self.nlags:, :, :]
                _pre_zt_p1_perm_flatten = _pre_zt_p1.permute(0, 2, 1).reshape(-1, self.est_zt_dim)
                zt_p1_perm_flatten = torch.cat([zt_p1_perm_flatten, _pre_zt_p1_perm_flatten], dim=-1)

        h_inv_zt_p1_perm_flatten = self.h_inv(zt_p1_perm_flatten)
        h_inv_zt_p1 = h_inv_zt_p1_perm_flatten.reshape(length-self.nlags, batch_size, self.xt_dim).permute(0, 2, 1)

        return zt_p1, f_zt_p1, h_m_xt_p1, h_inv_zt_p1
    
    def compute_h_xt(self, xt):
        length, xt_dim, batch_size = xt.shape

        xt_perm_flatten = xt.permute(0, 2, 1).reshape(-1, self.xt_dim)
        h_xt_perm_flatten = self.h(xt_perm_flatten)
        h_xt = h_xt_perm_flatten.reshape(length, batch_size, self.est_zt_dim).permute(0, 2, 1)

        return h_xt


class SeqEstARModel_CurrentPreZT(torch.nn.Module): 
    def __init__(self, nlags, xt_dim, est_zt_dim, hid_dim, n_pre_zt, dropout=0.5, poc_flag=False):
        super(SeqEstARModel_CurrentPreZT, self).__init__()

        self.nlags = nlags
        self.xt_dim = xt_dim
        self.est_zt_dim = est_zt_dim # the dim of the zt to estimate; assuming the hidden dim for each layer is the same
        self.hid_dim = hid_dim
        self.n_pre_zt = n_pre_zt 
        self.dropout = dropout
        self.poc_flag = poc_flag # this model can also be used as proof oc concept only, and can take gt_zt as input during training

        # define the mapping from xt to est_zt
        self.h = torch.nn.Sequential(
                torch.nn.Linear(self.xt_dim, self.hid_dim),
                torch.nn.BatchNorm1d(self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hid_dim, self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hid_dim, self.est_zt_dim)
            )
        
        # define the pseudo inverse of h_i
        self.h_inv = torch.nn.Sequential(
                    torch.nn.Linear(self.est_zt_dim * (self.n_pre_zt + 1), self.hid_dim),
                    torch.nn.BatchNorm1d(self.hid_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(self.dropout),
                    torch.nn.Linear(self.hid_dim, self.hid_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hid_dim, self.xt_dim)
            )
        
        # define the AR model on xt
        self.m = torch.nn.Sequential(
                torch.nn.Linear(self.xt_dim * self.nlags + self.est_zt_dim * self.n_pre_zt, self.hid_dim),
                torch.nn.BatchNorm1d(self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hid_dim, self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hid_dim, self.xt_dim)
            )

        # define the AR model on zt 
        self.f = torch.nn.Sequential(
                torch.nn.Linear(self.est_zt_dim * self.nlags + self.est_zt_dim * self.n_pre_zt, self.hid_dim),
                torch.nn.BatchNorm1d(self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hid_dim, self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hid_dim, self.est_zt_dim)
            )

    def forward(self, xt, pre_zt_list=None, zt=None):
        '''
        1. xt is a torch.tensor of shape: (seq_length, xt_dim, batch_size)
        2. pre_zt_list (optional; will not be given when estimate z1) is a list of previously estimated zt [(seq_length, xt_dim, batch_size),...], as each element is a tranformed xt on the xt_dim dimension 
        3. zt (optional) is a torch.tensor of shape: (seq_length, est_zt_dim, batch_size)6
        '''
        if pre_zt_list is None:
            pre_zt_list = []
        
        if zt is not None:
            self.poc_flag = True

        length, xt_dim, batch_size = xt.shape
        assert xt_dim == self.xt_dim
        assert xt_dim > self.est_zt_dim

        n_pre_zt = len(pre_zt_list)
        if n_pre_zt > 0:
            assert n_pre_zt == self.n_pre_zt
            for pre_zt in pre_zt_list:
                _length, _zt_dim, _batch_size = pre_zt.shape
                assert _length == length
                assert _batch_size == batch_size
                assert _zt_dim == self.est_zt_dim # assuming the dim of all zts are the same
        
        ### CI constraint: estimate z_{t+1} (zt_p1) from h, m ###
        # part 1: h(xt)
        xt_perm_flatten = xt.permute(0, 2, 1).reshape(-1, self.xt_dim)
        if not self.poc_flag:
            zt = self.h(xt_perm_flatten).reshape(length, batch_size, self.est_zt_dim).permute(0, 2, 1)

        # part 2: h * m (xt)
        m_xt_p1 = []
        for t in range(self.nlags, length):
            xt_history = xt[t-self.nlags:t, :, :]
            xt_history_perm_flatten = xt_history.permute(2, 1, 0).reshape(batch_size, self.xt_dim * self.nlags)
            # modified: add pre_zt_p1
            if self.n_pre_zt > 0:
                pre_zt_p1_perm_flatten = [pre_zt[t, :, :].permute(1, 0) for pre_zt in pre_zt_list]
                pre_zt_p1_perm_flatten = torch.cat(pre_zt_p1_perm_flatten, -1)
                xt_history_perm_flatten = torch.cat([pre_zt_p1_perm_flatten, xt_history_perm_flatten], dim=-1)
            m_xt_history = self.m(xt_history_perm_flatten).reshape(batch_size, self.xt_dim).permute(1, 0)
            m_xt_p1.append(m_xt_history)
        m_xt_p1 = torch.stack(m_xt_p1, dim=0) # length = length - nlags
        h_m_xt_p1_perm_flatten = self.h(m_xt_p1.permute(0, 2, 1).reshape(-1, self.xt_dim))
        h_m_xt_p1 = h_m_xt_p1_perm_flatten.reshape(length - self.nlags, batch_size, self.est_zt_dim).permute(0, 2, 1)

        ### CI constraint: estimate z_{t+1} (zt_p1) from f, h ###
        # part 1: reused
        # part 2: f * h (xt) = f(zt)
        f_zt_p1 = []
        for t in range(self.nlags, length):
            zt_history = zt[t-self.nlags:t, :, :]
            zt_history_perm_flatten = zt_history.permute(2, 0, 1).reshape(-1, self.est_zt_dim * self.nlags)
            if self.n_pre_zt > 0:
                zt_history_perm_flatten = torch.cat([zt_history_perm_flatten, pre_zt_p1_perm_flatten], dim=-1)
            f_zt_history = self.f(zt_history_perm_flatten).reshape(batch_size, self.est_zt_dim).permute(1, 0)
            f_zt_p1.append(f_zt_history)
        f_zt_p1 = torch.stack(f_zt_p1, dim=0) # length = length - nlags

        # MI constraint: reconstruct xt from zt_p1 and pre_zt_p1
        zt_p1 = zt[self.nlags:, :, :]
        zt_p1_perm_flatten = zt_p1.permute(0, 2, 1).reshape(-1, self.est_zt_dim)
        if self.n_pre_zt > 0:
            for pre_zt in pre_zt_list:
                _pre_zt_p1 = pre_zt[self.nlags:, :, :]
                _pre_zt_p1_perm_flatten = _pre_zt_p1.permute(0, 2, 1).reshape(-1, self.est_zt_dim)
                zt_p1_perm_flatten = torch.cat([zt_p1_perm_flatten, _pre_zt_p1_perm_flatten], dim=-1)

        h_inv_zt_p1_perm_flatten = self.h_inv(zt_p1_perm_flatten)
        h_inv_zt_p1 = h_inv_zt_p1_perm_flatten.reshape(length-self.nlags, batch_size, self.xt_dim).permute(0, 2, 1)

        return zt_p1, f_zt_p1, h_m_xt_p1, h_inv_zt_p1
    
    def compute_h_xt(self, xt):
        length, xt_dim, batch_size = xt.shape

        xt_perm_flatten = xt.permute(0, 2, 1).reshape(-1, self.xt_dim)
        h_xt_perm_flatten = self.h(xt_perm_flatten)
        h_xt = h_xt_perm_flatten.reshape(length, batch_size, self.est_zt_dim).permute(0, 2, 1)

        return h_xt
    
