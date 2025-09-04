import os
import torch
import random
import argparse

# Simple AR model for proof of concept
class SimpleARModel(torch.nn.Module): # three layer by default
    def __init__(self, nlags, in_dim, hid_dim, dropout=0.5):
        super(SimpleARModel, self).__init__()

        self.nlags = nlags
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.dropout = dropout

        self.Bs = torch.nn.ModuleList()
        for lag in range(self.nlags):
            net = torch.nn.Sequential(
                torch.nn.Linear(self.in_dim, self.hid_dim),
                torch.nn.BatchNorm1d(self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hid_dim, self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hid_dim, self.in_dim)
            )
            self.Bs.append(net)

    def forward(self, zt):
        length, nlatents, batch_size = zt.shape
        zt_recon = []
        for t in range(self.nlags, length):
            _zt_recon = 0
            for lag in range(self.nlags):
                _zt_recon += self.Bs[lag](zt[t-lag].T)
            zt_recon.append(_zt_recon)
        zt_recon = torch.stack(zt_recon, 0)

        return zt_recon
    
# Sequentially estimated model
class SeqEstARModel(torch.nn.Module): 
    def __init__(self, nlags, xt_dim, est_zt_dim, hid_dim, n_pre_zt, dropout=0.5):
        super(SeqEstARModel, self).__init__()

        self.nlags = nlags
        self.xt_dim = xt_dim
        self.est_zt_dim = est_zt_dim # the dim of the zt to estimate; assuming the hidden dim for each layer is the same
        self.hid_dim = hid_dim
        self.n_pre_zt = n_pre_zt 
        self.dropout = dropout

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
                    torch.nn.Linear(self.est_zt_dim*(self.n_pre_zt+1), self.hid_dim),
                    torch.nn.BatchNorm1d(self.hid_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(self.dropout),
                    torch.nn.Linear(self.hid_dim, self.hid_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hid_dim, self.xt_dim)
            )
        
        # define the AR model on xt
        self.m = torch.nn.ModuleDict()
        for lag in range(self.nlags):
            net_m = torch.nn.Sequential(
                torch.nn.Linear(self.xt_dim, self.hid_dim),
                torch.nn.BatchNorm1d(self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hid_dim, self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hid_dim, self.xt_dim)
            )
            self.m[f"lag_{lag}"] = net_m

        # define the AR model on zt 
        self.f = torch.nn.ModuleDict()
        for lag in range(self.nlags):
            net_f = torch.nn.Sequential(
                torch.nn.Linear(self.est_zt_dim*(self.n_pre_zt+1), self.hid_dim),
                torch.nn.BatchNorm1d(self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hid_dim, self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hid_dim, self.est_zt_dim)
            )
            self.f[f"lag_{lag}"] = net_f

    def forward(self, xt, pre_zt_list=None):
        '''
        1. xt is a torch.tensor of shape: (seq_length, xt_dim, batch_size)
        2.1. pre_zt is a list of previously estimated zt [(seq_length, xt_dim, batch_size),...], as each element is a tranformed xt on the xt_dim dimension 
        2.2. pre_zt is not given when estimate z1
        '''
        if pre_zt_list is None:
            pre_zt_list = []
        length, xt_dim, batch_size = xt.shape
        assert xt_dim == self.xt_dim
        assert xt_dim > self.est_zt_dim

        n_pre_zt = len(pre_zt_list)
        if n_pre_zt > 0:
            assert n_pre_zt == self.n_pre_zt
            for zt in pre_zt_list:
                _length, _zt_dim, _batch_size = zt.shape
                assert _length == length
                assert _batch_size == batch_size
                assert _zt_dim == self.est_zt_dim # assuming the dim of all zts are the same
        
        ### CI constraint: estimate z_{t+1} (zt_p1) from h, m ###
        # part 1: h(xt)
        xt_perm_flatten = xt.permute(0, 2, 1).reshape(-1, self.xt_dim)
        h_xt_perm_flatten = self.h(xt_perm_flatten)
        h_xt = h_xt_perm_flatten.reshape(length, batch_size, self.est_zt_dim).permute(0, 2, 1)

        # part 2: h * m (xt)
        m_xt_p1 = []
        for t in range(self.nlags, length):
            m_xt_p1_t = 0
            for lag in range(self.nlags):
                _xt = xt[t-lag, :, :]
                _xt_perm = _xt.permute(1, 0)
                _m_xt = self.m[f'lag_{lag}'](_xt_perm)
                m_xt_p1_t += _m_xt.permute(1, 0)
            m_xt_p1.append(m_xt_p1_t)
        m_xt_p1 = torch.stack(m_xt_p1, dim=0) # length = length - nlags
        h_m_xt_p1_perm_flatten = self.h(m_xt_p1.permute(0, 2, 1).reshape(-1, xt_dim))
        h_m_xt_p1 = h_m_xt_p1_perm_flatten.reshape(length - self.nlags, batch_size, self.est_zt_dim).permute(0, 2, 1)

        ### CI constraint: estimate z_{t+1} (zt_p1) from f, h ###
        # part 1: reused
        # part 2: f * h (xt) = f(zt)
        f_zt_p1 = []
        for t in range(self.nlags, length):
            f_zt_p1_t = 0
            for lag in range(self.nlags):
                _zt = h_xt[t-lag, :, :]
                if self.n_pre_zt > 0:
                    for pre_zt in pre_zt_list:
                        _pre_zt = pre_zt[t-lag, :, :]
                        _zt = torch.cat([_zt, _pre_zt], dim=0)
                _zt_perm = _zt.permute(1, 0)
                _f_zt = self.f[f'lag_{lag}'](_zt_perm)
                f_zt_p1_t += _f_zt.permute(1, 0)
            f_zt_p1.append(f_zt_p1_t)
        f_zt_p1 = torch.stack(f_zt_p1, dim=0) # length = length - nlags

        # MI constraint: reconstruct xt from zt_p1 and pre_zt_p1
        zt_p1 = h_xt[self.nlags:, :, :]
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
