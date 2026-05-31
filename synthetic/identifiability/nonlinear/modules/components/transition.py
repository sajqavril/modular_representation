"""Prior Network"""
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
from torch.func import jacfwd, vmap
from .mlp import NLayerLeakyMLP, NLayerLeakyNAC
from .base import GroupLinearLayer
try:
    import ipdb as pdb
except ImportError:
    pdb = None

class NPTransitionPrior(nn.Module):

    def __init__(
        self, 
        lags, 
        latent_size, 
        num_layers=3, 
        hidden_dim=64):
        super().__init__()
        self.L = lags
        # self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))       
        gs = [NLayerLeakyMLP(in_features=lags*latent_size+1, 
                             out_features=1, 
                             num_layers=num_layers, 
                             hidden_dim=hidden_dim) for i in range(latent_size)]
        
        self.gs = nn.ModuleList(gs)
    
    def forward(self, x, masks=None):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        batch_size, length, input_dim = x.shape
        # init_hiddens = self.init_hiddens.repeat(batch_size, 1, 1)
        # x = torch.cat((init_hiddens, x), dim=1)
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape
        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        yy = yy.reshape(-1, self.L*input_dim)
        residuals = [ ]

        hist_jac = []

        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            if masks is None:
                inputs = torch.cat((yy, xx[:,:,i]),dim=-1)
            else:
                mask = masks[i]
                inputs = torch.cat((yy*mask, xx[:,:,i]),dim=-1)
            residual = self.gs[i](inputs)
            with torch.enable_grad():
            #     pdd = jacobian(self.gs[i], inputs, create_graph=True, vectorize=True)
            # # Determinant of low-triangular mat is product of diagonal entries
            # logabsdet = torch.log(torch.abs(torch.diag(pdd[:,0,:,-1])))
                pdd = vmap(jacfwd(self.gs[i]))(inputs)
            logabsdet = torch.log(torch.abs(pdd[:,0,-1]))
            hist_jac.append(torch.unsqueeze(pdd[:,0,:-1], dim=1))
            # replaced with faster jac (up)
            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian, hist_jac

class NPChangeTransitionPrior(nn.Module):


    def __init__(
        self, 
        lags, 
        latent_size,
        embedding_dim, 
        num_layers=3,
        hidden_dim=64):
        super().__init__()
        self.L = lags
        # self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))       
        gs = [NLayerLeakyMLP(in_features=hidden_dim+lags*latent_size+1, 
                             out_features=1, 
                             num_layers=0, 
                             hidden_dim=hidden_dim) for i in range(latent_size)]
        
        self.gs = nn.ModuleList(gs)
        self.fc = NLayerLeakyMLP(in_features=embedding_dim,
                                 out_features=hidden_dim,
                                 num_layers=2,
                                 hidden_dim=hidden_dim)

    def forward(self, x, embeddings, masks=None):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        # embeddings: [BS, embed_dims]
        batch_size, length, input_dim = x.shape
        # import pdb
        # pdb.set_trace()
        embeddings = self.fc(embeddings)
        embeddings = embeddings.unsqueeze(1).repeat(1, length-self.L, 1).reshape(-1, embeddings.shape[-1])
        # init_hiddens = self.init_hiddens.repeat(batch_size, 1, 1)
        # x = torch.cat((init_hiddens, x), dim=1)
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape
        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        yy = yy.reshape(-1, self.L*input_dim)
        residuals = [ ]
        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            if masks is None:
                inputs = torch.cat((embeddings, yy, xx[:,:,i]),dim=-1)
            else:
                mask = masks[i]
                inputs = torch.cat((embeddings, yy*mask, xx[:,:,i]),dim=-1)
            residual = self.gs[i](inputs)
            with torch.enable_grad():
                # pdd = jacobian(self.gs[i], inputs, create_graph=True, vectorize=True)
                pdd = vmap(jacfwd(self.gs[i]))(inputs)
            # Determinant of low-triangular mat is product of diagonal entries
            logabsdet = torch.log(torch.abs(pdd[:,0,-1]))
            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian
    
class NPChangeInstantaneousTransitionPrior(nn.Module):

    def __init__(
        self, 
        lags, 
        latent_size,
        embedding_dim, 
        num_layers=3,
        hidden_dim=64):
        super().__init__()
        self.L = lags
        # self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))       
        gs = [NLayerLeakyMLP(in_features=hidden_dim+lags*latent_size+1+i, 
                             out_features=1, 
                             num_layers=0, 
                             hidden_dim=hidden_dim) for i in range(latent_size)]
        
        self.gs = nn.ModuleList(gs)
        self.fc = NLayerLeakyMLP(in_features=embedding_dim,
                                 out_features=hidden_dim,
                                 num_layers=2,
                                 hidden_dim=hidden_dim)

    def forward(self, x, embeddings, alphas):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        # embeddings: [BS, embed_dims]
        batch_size, length, input_dim = x.shape
        embeddings = self.fc(embeddings)
        embeddings = embeddings.unsqueeze(1).repeat(1, length-self.L, 1).reshape(-1, embeddings.shape[-1])
        # prepare data
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape
        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        yy = yy.reshape(-1, self.L*input_dim)
        # get residuals and |J|
        residuals = [ ]
        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            inputs = torch.cat([xx[:,:,j] * alphas[i][j] for j in range(i)] + [embeddings, yy, xx[:,:,i]], dim=-1)
            residual = self.gs[i](inputs)
            with torch.enable_grad():
                pdd = vmap(jacfwd(self.gs[i]))(inputs)
            # Determinant: product of diagonal entries, sum of last entry
            logabsdet = torch.log(torch.abs(pdd[:,0,-1]))
            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian
 
class NPInstantaneousTransitionPrior(nn.Module):

    def __init__(
        self,
        lags,
        latent_size,
        num_layers=3,
        hidden_dim=64,
        layer_dims=None):
        """Instantaneous transition prior with block-lower lagged inputs.

        ``layer_dims`` defines contiguous latent blocks ordered from upper to
        lower layers. For a destination latent in layer k, the history term can
        only use lagged source latents from layers 0..k. This makes the
        transition Jacobian block lower triangular in the standard convention
        rows=to/destination and columns=from/source.
        """
        super().__init__()
        self.L = lags
        self.latent_size = latent_size
        if layer_dims is None:
            layer_dims = [latent_size]
        self.layer_dims = list(layer_dims)
        assert sum(self.layer_dims) == latent_size, (
            "sum(layer_dims)=%d must equal latent_size=%d"
            % (sum(self.layer_dims), latent_size))

        self.layer_ids = []
        for layer_id, dim in enumerate(self.layer_dims):
            self.layer_ids.extend([layer_id] * dim)

        self.allowed_hist_indices = []
        gs = []
        for i in range(latent_size):
            dst_layer = self.layer_ids[i]
            allowed_sources = [
                j for j, src_layer in enumerate(self.layer_ids)
                if src_layer <= dst_layer
            ]
            allowed_hist = [
                lag * latent_size + j
                for lag in range(lags)
                for j in allowed_sources
            ]
            self.allowed_hist_indices.append(allowed_hist)
            gs.append(NLayerLeakyMLP(in_features=len(allowed_hist)+1+i,
                                     out_features=1,
                                     num_layers=num_layers,
                                     hidden_dim=hidden_dim))

        self.gs = nn.ModuleList(gs)

    def forward(self, x):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        batch_size, length, input_dim = x.shape
        assert input_dim == self.latent_size

        x = x.unfold(dimension=1, size=self.L+1, step=1)
        x = torch.swapaxes(x, 2, 3)
        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:, -1:], x[:, :-1]
        yy = yy.reshape(-1, self.L * input_dim)

        residuals = []
        hist_jac = []
        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            allowed_hist = self.allowed_hist_indices[i]
            hist_inputs = yy[:, allowed_hist]
            inputs = torch.cat(
                [hist_inputs] + [xx[:, :, j] for j in range(i)] + [xx[:, :, i]],
                dim=-1)

            residual = self.gs[i](inputs)
            with torch.enable_grad():
                pdd = vmap(jacfwd(self.gs[i]))(inputs)

            # Determinant: product of diagonal entries, sum of last entry.
            logabsdet = torch.log(torch.abs(pdd[:, 0, -1]))
            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

            # Return a full-width Jacobian for plotting/sparsity code.
            # Forbidden lagged source blocks remain exactly zero.
            n_hist = len(allowed_hist)
            full_jac = inputs.new_zeros(inputs.shape[0], self.L * input_dim + i)
            full_jac[:, allowed_hist] = pdd[:, 0, :n_hist]
            if i > 0:
                full_jac[:, self.L * input_dim:] = pdd[:, 0, n_hist:-1]
            hist_jac.append(torch.unsqueeze(full_jac, dim=1))

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(
            sum_log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian, hist_jac
