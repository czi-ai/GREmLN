import torch
from torch_geometric.utils import scatter, remove_self_loops
from torch_geometric.data import Data, Batch
from scGraphLLM._globals import *  ## imported global variables are all caps 


def _identity(x):
    return x

def _exp_kernel(x, beta):
    return torch.exp(-beta * (x + 1))

def _cosine_kernel(x):
    PI = torch.acos(torch.Tensor([-1]))
    return torch.cos(PI * x / 2)

def _rescaled_L(edge_index, num_nodes, edge_weight=None):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight) 
    # DANGER
    if edge_index.shape[-1] == 0:
        idx = torch.arange(num_nodes, device=edge_index.device)
        edge_index = idx.unsqueeze(0).repeat(2, 1)
    # DANGER
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)
    row, col = edge_index[0], edge_index[1]

    if row.shape != edge_weight.shape:
        print(f"Shape mismatch: row={row.shape}, edge_weight={edge_weight.shape}")
    max_index = max_index = row.max().item()
    if max_index >= num_nodes:
        print(f"Row index out of bounds! max index = {max_index} >= num_nodes = {num_nodes}")
    if torch.isnan(edge_weight).any():
        print("NaN detected in edge_weight!")
    if torch.isinf(edge_weight).any():
        print("Inf detected in edge_weight!")

    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')
    deg = deg.clamp(min=1e-8)
    assert not torch.isnan(edge_weight).any(), "NaN values in edge_weight"
    assert not torch.isnan(deg).any(), "NaN values in degree"
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt.isnan(), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col] # D^(-1/2) * A * D^(-1/2)
    assert not torch.isnan(edge_weight).any(), "NaN values in edge_weight after normalization"
    L_rescaled = torch.sparse_coo_tensor(edge_index, -edge_weight, (num_nodes, num_nodes))
    return L_rescaled

# Cache for Chebyshev coefficients - only depends on K, beta, and device
_chebyshev_coeff_cache = {}

def _get_cached_chebyshev_coeff(K, beta, device, N=100):
    """
    Get cached Chebyshev coefficients. These only depend on K and beta,
    so we can cache them to avoid recomputation every forward pass.
    """
    cache_key = (K, beta, str(device))
    
    if cache_key not in _chebyshev_coeff_cache:
        ind = torch.arange(0, K+1, dtype=torch.float32, device=device)
        ratio = torch.pi * (torch.arange(1, N+1, dtype=torch.float32, device=device) - 0.5) / N
        x = torch.cos(ratio)  # quadrature points
        T_kx = torch.cos(ind.view(-1, 1) * ratio) 
        w = torch.ones(N, device=device) * (torch.pi / N)
        f_x = _exp_kernel(x, beta)
        c_k = (2 / torch.pi) * torch.matmul(T_kx, w * f_x)
        _chebyshev_coeff_cache[cache_key] = c_k
    
    return _chebyshev_coeff_cache[cache_key]

def _chebyshev_recurrence(L_csr, E_reshaped, c_k, K):
    """
    Core Chebyshev recurrence loop - separated for torch.compile optimization.
    """
    T_0 = E_reshaped
    T_1 = torch.sparse.mm(L_csr, E_reshaped)
    y = c_k[0] * T_0 + c_k[1] * T_1
    
    # start recursion
    T_k_prev = T_1
    T_k_prev_prev = T_0
    for i in range(2, K + 1):
        T_k = 2 * torch.sparse.mm(L_csr, T_k_prev) - T_k_prev_prev
        y = y + c_k[i] * T_k
        
        # shift index
        T_k_prev_prev = T_k_prev 
        T_k_prev = T_k
    
    return y

@torch.amp.autocast(enabled=False, device_type='cuda')
def _chebyshev_diffusion_batch(edge_index, num_nodes, E, k=128, edge_weight=None, beta=0.5):
    """
    E: (S, H, d)
    """
    L_rescaled = _rescaled_L(edge_index, num_nodes, edge_weight)
    
    # Convert to CSR format for faster sparse matrix multiplication
    L_csr = L_rescaled.to_sparse_csr()
    
    # Use cached coefficients (avoids recomputation every forward pass)
    c_k = _get_cached_chebyshev_coeff(k, beta, E.device)
    
    E = E.to(torch.float32)
    s, h, d = E.size()
    assert s == num_nodes, f"Expect {num_nodes} nodes, Got {s}"
    E_reshaped = E.reshape(num_nodes, h * d)
    c_k = c_k.to(torch.float32)
    
    y = _chebyshev_recurrence(L_csr, E_reshaped, c_k, k)
    
    final_emb = y.reshape(num_nodes, h, d)
    final_emb = final_emb.bfloat16()
    
    return final_emb

def _chebyshev_diffusion(edge_index_list, num_nodes_list, E, k=64, beta=0.5):
    """
    edge index list: list of edge index, length B
    E: (B, S, H, d)
    """
    B, S, H, D = E.size()
    
    # 1. Create a mask to un-pad the (B, S, H, D) tensor
    num_nodes_tensor = torch.tensor(num_nodes_list, device=E.device, dtype=torch.long)
        
    # Create a (B, S) boolean mask
    # Broadcasting creates a (B, S) matrix of comparisons
    mask = torch.arange(S, device=E.device)[None, :] < num_nodes_tensor[:, None]
    
    # 2. Un-pad E
    # Use the boolean mask to select only the "real" node embeddings
    # Shape goes from (B, S, H, D) -> (total_nodes, H, D)
    unpadded_E = E[mask]
    
    # 3. Batch edge indices with offsets (avoids PyG Batch.from_data_list overhead)
    offsets = torch.zeros(B, dtype=torch.long, device=E.device)
    if B > 1:
        offsets[1:] = num_nodes_tensor[:-1].cumsum(0)
    
    batched_edges = []
    for i, ei in enumerate(edge_index_list):
        ei_device = ei.to(E.device) if ei.device != E.device else ei
        batched_edges.append(ei_device + offsets[i])
    batched_edge_index = torch.cat(batched_edges, dim=1)
    total_nodes = num_nodes_tensor.sum().item()

    # 4. Run batched diffusion
    # Pass the entire batch of nodes and the single batched edge_index
    diffused_unpadded_E = _chebyshev_diffusion_batch(
        batched_edge_index, 
        total_nodes, 
        unpadded_E, 
        k=k, 
        edge_weight=None, # Assuming edge_weight is None as per GDTransformer's call
        beta=beta
    )
    
    # 5. Re-pad the result
    # Create a zero tensor with the original padded shape
    final_emb = torch.zeros_like(E)
    
    # Use the mask to "scatter" the diffused embeddings back to their
    # original positions in the padded tensor
    final_emb[mask] = diffused_unpadded_E

    
    assert final_emb.size() == E.size(), f"Expect {E.size()}, Got {final_emb.size()}"
    return final_emb