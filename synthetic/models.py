import os
import torch
import random
import argparse

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
                _zt_recon += self.Bs[lag](zt[t-lag])
            zt_recon.append(_zt_recon)
        zt_recon = torch.stack(zt_recon, 0)

        return zt_recon
    

class ComplexARModel(torch.nn.Module): # three layer by default
    def __init__(self, nlags, in_dim_zt, in_dim_xt, hid_dim, dropout=0.5):
        super(ComplexARModel, self).__init__()

        self.nlags = nlags
        self.in_dim_zt = in_dim_zt
        self.in_dim_xt = in_dim_xt
        self.hid_dim = hid_dim
        self.dropout = dropout

        # define two ARModels on zt(1) and xt, respectively
        self.Bs_zt = torch.nn.ModuleList()
        self.Bs_xt = torch.nn.ModuleList()
        for lag in range(self.nlags):
            net_zt = torch.nn.Sequential(
                torch.nn.Linear(self.in_dim_zt, self.hid_dim),
                torch.nn.BatchNorm1d(self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hid_dim, self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hid_dim, self.in_dim_zt)
            )
            net_xt = torch.nn.Sequential(
                torch.nn.Linear(self.in_dim_xt, self.hid_dim),
                torch.nn.BatchNorm1d(self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hid_dim, self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hid_dim, self.in_dim_xt)
            )
            self.Bs_zt.append(net_zt)
            self.Bs_xt.append(net_xt)
        
        # the function from x to z(1) for H(zt+1|zt)
        self.g_Hzt = torch.nn.Sequential(
                torch.nn.Linear(self.in_dim_xt, self.hid_dim),
                torch.nn.BatchNorm1d(self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hid_dim, self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hid_dim, self.in_dim_zt)
            )
        
        # the function from x to z(1) for H(zt+1|xt)
        self.g_Hxt = torch.nn.Sequential(
                torch.nn.Linear(self.in_dim_xt, self.hid_dim),
                torch.nn.BatchNorm1d(self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout),
                torch.nn.Linear(self.hid_dim, self.hid_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hid_dim, self.in_dim_zt)
            )

    def forward(self, xt):
        length, xt_dim, batch_size = xt.shape
        assert xt_dim == self.in_dim_xt

        ### Estimate zt+1 from zt ### 
        # compute zt from xt by g_Hzt
        xt_perm = xt.permute(0, 2, 1)
        xt_flatten = xt_perm.reshape(-2, xt_dim)
        zt = self.g_Hzt(xt_flatten).reshape(length, batch_size, self.in_dim_zt).perm(0, 2, 1)

        # compute recon_zt from zt by Bs_zt
        zt_recon_Hzt = []
        for t in range(self.nlags, length):
            _zt_recon = 0
            for lag in range(self.nlags):
                _zt_recon += self.Bs_zt[lag](zt[t-lag])
            zt_recon_Hzt.append(_zt_recon)
        zt_recon_Hzt = torch.stack(zt_recon_Hzt, 0)

        ### Estimate zt+1 from xt ###
        # compute recon_xt from xt by Bs_xt
        xt_recon = []
        for t in range(self.nlags, length):
            _xt_recon = 0
            for lag in range(self.nlags):
                _xt_recon += self.Bs_xt[lag](xt[t-lag])
            xt_recon.append(_xt_recon)
        xt_recon = torch.stack(xt_recon, 0)

        # compute zt from xt_recon by g_Hxt
        xt_recon_perm = xt_recon.permute(0, 2, 1)
        xt_recon_flatten = xt_recon_perm.reshape(-2, xt_dim)
        zt_recon_Hxt = self.g_Hxt(xt_recon_flatten).reshape(length, batch_size, self.in_dim_zt).perm(0, 2, 1)

        return xt_recon, zt_recon, zt_recon_Hzt, zt_recon_Hxt
