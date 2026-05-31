import torch
import torch.nn as nn 

class LowerTriangularHierarchicalMatrices(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims)

        rows = []
        for i in range(self.n_layers):
            row = nn.ParameterList()
            for j in range(i):
                row.append(
                    nn.Parameter(torch.randn(layer_dims[i], layer_dims[j]))
                )
            rows.append(row)
        self.C = nn.ModuleList(rows)

    def __getitem__(self, idx):
        i, j = idx
        if j >= i:
            raise IndexError("Only lower-triangular connections exist: require j < i")
        return self.C[i][j]
    
    def get_all_matrices(self):
        """Return a list of lists of matrices, where the (i,j)-th entry is C[i,j] if j < i, else None"""
        all_matrices = []
        for i in range(self.n_layers):
            for j in range(self.n_layers):
                if j < i:
                    all_matrices.append(self.C[i][j])
        return all_matrices
    
    
def reorganize_instan_matrices(A_lists):
    # A_lists is a list of L matrices, where A_lists[l] is the (z_dim_l, z_dim_l) matrix for layer l.
    # We want to create a single block-diagonal matrix A_full of shape (sum(z_dims), sum(z_dims)),
    # where the l-th block on the diagonal is A_lists[l].
    z_dims = [A.shape[0] for A in A_lists]
    z_total = sum(z_dims)
    
    A_full = torch.zeros((z_total, z_total), device=A_lists[0].device)
    
    start = 0
    for l, A in enumerate(A_lists):
        d = z_dims[l]
        A_full[start:start+d, start:start+d] = A
        start += d
    
    return A_full

def reorganize_trans_matrices(B_lists, C_lists):
    # B_lists is a list of L matrices, where B_lists[l]is the (z_dim_l, z_dim_l) transition matrix for layer l.
    # C_lists is a matrix of shape (L, L) where C_lists[l, j] is the (z_dim_l, z_dim_j) cross-layer transition matrix from layer j to layer l (l > j)
    
    z_dims = [B.shape[0] for B in B_lists]
    z_total = sum(z_dims)
    
    B_full = torch.zeros((z_total, z_total), device=B_lists[0].device)
    
    start = 0
    for l, B in enumerate(B_lists):
        d = z_dims[l]
        B_full[start:start+d, start:start+d] = B
        start += d
    
    start_1 = 0
    start_2 = 0
    for l, b in enumerate(B_lists):
        d_1 = z_dims[l]
        for j in range(l):
            C = C_lists[l, j]
            d_2 = z_dims[j]

            B_full[start_1:start_1+d_1, start_2:start_2+d_2] = C
            start_2 += d_2
        start_1 += d_1
        start_2 = 0
        
    return B_full    