import torch
import torch.nn as nn
from utils.model_utils import LowerTriangularHierarchicalMatrices, reorganize_instan_matrices

class LinearHierarchicalTemporalICA(nn.Module):
    def __init__(self, x_dim, z_dims,):
        super().__init__()
        self.x_dim = x_dim
        self.z_dims = z_dims
        self.A_lists = nn.ParameterList([
            nn.Parameter(torch.zeros((d, d)), requires_grad=False) for d in z_dims
        ])
        self.B_lists = nn.ParameterList([
            nn.Parameter(torch.randn((d, d))) for d in z_dims
        ])
        self.C_lists = LowerTriangularHierarchicalMatrices(z_dims)
        self.M_lists = nn.ParameterList([
            nn.Parameter(torch.randn((x_dim, d))) for d in z_dims
        ])
        
    def encoder_matrix(self):
        return torch.cat([torch.pinverse(M_l).T for M_l in self.M_lists], dim=1)
    
    def decoder_matrix(self):
        return torch.cat([M_l for M_l in self.M_lists], dim=1)
        
    def encode_layers(self, x):
        z_layers = []
        for M_l in self.M_lists:
            M_l_pinv = torch.pinverse(M_l)
            z_layers.append(torch.matmul(x, M_l_pinv.T))
        return z_layers
    
    def encode(self, x):
        return torch.cat(self.encode_layers(x), dim=-1)
    
    def decode_layers(self, z_layers):
        x_hat = z_layers[0].new_zeros(z_layers[0].shape[:-1] + (self.x_dim,))
        for z_l, M_l in zip(z_layers, self.M_lists):
            x_hat = x_hat + torch.matmul(z_l, M_l.T)
        return x_hat
    
    def decode(self, z):
        return self.decode_layers(self.z_to_layers(z))
    
    def z_to_layers(self, z):
        return list(torch.split(z, self.z_dims, dim=-1))
    
    def estimate_noise(self, z):
        z = self.z_to_layers(z)
        epss = []
        for i in range(len(self.z_dims)):
            z_l = z[i]
            z_l_t_now = z_l[:,0,:]
            z_l_t_next = z_l[:,1,:]
            A_l = self.A_lists[i]
            B_l = self.B_lists[i]
            
            I_A_l = torch.eye(A_l.shape[0], device=A_l.device, dtype=A_l.dtype) - A_l
            z_t1_I_A_l = torch.matmul(z_l_t_next, I_A_l.T)
            cross_layer_contribution = torch.zeros_like(z_l_t_next)
            for j in range(i):
                cross_layer_contribution += torch.matmul(z[j][:,0,:], self.C_lists[i,j].T)
            eps = z_t1_I_A_l - torch.matmul(z_l_t_now, B_l.T) - cross_layer_contribution
            epss.append(eps)
        return epss
    
    def forward(self, x):
        z_hat = self.encode(x)
        epss_hat = self.estimate_noise(z_hat)
        x_hat = self.decode(z_hat)
        
        epss_hat = torch.cat(epss_hat, dim=1)  # (batch_size, sum(z_dims))
        
        return {
            "x_hat": x_hat,
            "z_hat": z_hat,
            "epss_hat": epss_hat,
        }
        

def calc_ica_loss(model, epss_hat):
    A = reorganize_instan_matrices(model.A_lists)
    I_A = torch.eye(A.shape[0], device=A.device, dtype=A.dtype) - A
    encoder = model.encoder_matrix()
    tmp = torch.matmul(encoder, I_A.T)
    
    if tmp.shape[0] == tmp.shape[1]:
        jacob_loss = -torch.slogdet(tmp)[1]
    else:
        gram = torch.matmul(tmp.T, tmp)
        ridge = 1e-6 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        jacob_loss = -0.5 * torch.slogdet(gram + ridge)[1]
    
    dist_loss = torch.mean(torch.abs(epss_hat))
    ica_loss = dist_loss + jacob_loss
    
    return ica_loss
    
        
        
