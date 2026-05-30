import torch
import numpy as np
from scipy.io import savemat
import os


def _ensure_col(z, M):
    if z is None: return None
    z = np.asarray(z)
    if z.ndim == 0: z = np.full((M,1), float(z))
    elif z.ndim == 1: z = z.reshape(-1,1)
    return torch.tensor(z, dtype=torch.float32)

def _to_tensor(x): return None if x is None else torch.tensor(np.asarray(x), dtype=torch.float32)

def edge_lengths(pos, edge_index):
    src, dst = edge_index
    vec = pos[dst] - pos[src]
    return torch.linalg.norm(vec, dim=1, keepdim=True), vec

def assemble_bc(num_nodes):
    fixed = torch.zeros((num_nodes,2), dtype=torch.bool)
    fixed[0,:] = True  # b0 fixed (x,y)
    fixed[3,:] = True  # t0 fixed (x,y)
    return fixed

def rel_l2_stats_batch(
    pred: torch.Tensor,         # [N, 2]
    target: torch.Tensor,       # [N, 2]
    fixed_mask: torch.Tensor,   # [N, 2] bool (True=fixed) or None
    batch_index: torch.Tensor,  # [N] graph id per node (0..G-1)
    eps: float = 1e-12,
):
    """
    Compute per-graph relative L2 over FREE DOFs only, then return
    sufficient statistics across the graphs in this batch:
        sum(rel_i), sum(rel_i^2), count

    rel_i = ||(pred - target)_free||_2 / (||target_free||_2 + eps)

    Graphs with ||target_free||_2 < eps (degenerate targets) are skipped.
    """
    if fixed_mask is None:
        fixed_mask = torch.zeros_like(target, dtype=torch.bool)

    if batch_index.numel() == 0:
        return 0.0, 0.0, 0

    G = int(batch_index.max().item()) + 1
    sum_rel, sumsq_rel, cnt = 0.0, 0.0, 0

    for g in range(G):
        nodes = (batch_index == g)
        if nodes.sum() == 0:
            continue

        m = ~fixed_mask[nodes]
        if m.sum() == 0:
            continue

        err = (pred[nodes] - target[nodes])[m].to(torch.float64)
        tru = target[nodes][m].to(torch.float64)

        den = torch.linalg.norm(tru)
        if den < eps:
            continue

        rel = (torch.linalg.norm(err) / (den + eps)).item()
        sum_rel += rel
        sumsq_rel += rel * rel
        cnt += 1

    return sum_rel, sumsq_rel, cnt


def rel_l2_stats_batch_masked(pred, target, batch_index, row_mask, eps=1e-12):
    """
    Compute per-graph relative L2 = ||pred - target|| / ||target||,
    but only over rows where row_mask is True. Then aggregate mean & std.

    Args:
        pred:   (N, d)
        target: (N, d)
        batch_index: (N,) graph indices (PyG's batch vector)
        row_mask: (N,) boolean mask for which nodes to include
        eps: small number to avoid divide-by-zero

    Returns:
        s: sum of per-graph rel-L2
        ssq: sum of squares of per-graph rel-L2
        c: number of graphs that had at least one masked row
    """
    device = pred.device
    num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0

    s = 0.0
    ssq = 0.0
    c = 0

    for g in range(num_graphs):
        g_rows = (batch_index == g) & row_mask
        if not torch.any(g_rows):
            continue

        e = pred[g_rows] - target[g_rows]          # (Ng, d)
        num = torch.linalg.norm(e).item()
        den = torch.linalg.norm(target[g_rows]).item()
        rel = num / max(den, eps)

        s += rel
        ssq += rel * rel
        c += 1

    return s, ssq, c



def _unscale_y(pred_scaled, y_scaled, scale_y, batch_index):
    """
    Unscale node-wise using per-graph scalar `scale_y`.
    Handles:
      - scale_y shape [] (scalar, one graph)
      - scale_y shape [G] or [G,1] (batched graphs)
    """
    if scale_y.ndim == 0:
        s = scale_y.view(1)  # single graph
        return pred_scaled / s, y_scaled / s

    # scale_y is [G] or [G,1]
    s = scale_y.view(-1)  # [G]
    # broadcast to nodes via batch_index
    s_nodes = s[batch_index].unsqueeze(-1)  # [N,1] to match [N,2]
    return pred_scaled / s_nodes, y_scaled / s_nodes


def _unscale_y_planeframe(pred_scaled: torch.Tensor,
               y_scaled: torch.Tensor,
               scale_y,
               batch_index: torch.Tensor):
    """
    Unscale node-wise (u, v, rot).

    Accepts:
      pred_scaled, y_scaled: [N,3] or [B,N,3]
      batch_index:           [N]   or [B,N]   (graph ids per node, starting at 0)
      scale_y (shared across batch):
        - scalar                     -> []
        - per-component              -> [3] or [1,3]
        - per-graph scalar           -> [G] or [G,1]
        - per-graph per-component    -> [G,3]
      (Optional) per-batch versions are allowed as [B, G, 1/3].

    Returns tensors with the same shape as inputs.
    """
    assert pred_scaled.shape == y_scaled.shape and pred_scaled.size(-1) == 3
    device, dtype = pred_scaled.device, pred_scaled.dtype

    # --- Normalize inputs to [B,N,3] and batch_idx to [B,N] ---
    if pred_scaled.ndim == 2:
        pred = pred_scaled.unsqueeze(0)   # [1,N,3]
        y    = y_scaled.unsqueeze(0)
        batch_idx = batch_index.unsqueeze(0)  # [1,N]
        squeeze_back = True
    else:
        pred = pred_scaled                # [B,N,3]
        y    = y_scaled
        batch_idx = batch_index           # [B,N]
        squeeze_back = False

    B, N, D = pred.shape
    if N == 0:
        return pred_scaled, y_scaled

    # --- Determine G (graphs per batch item) upper bound ---
    Gmax = int(batch_idx.max().item()) + 1

    # --- Normalize scale_y to [B', G', K] (K in {1,3}) ---
    s = torch.as_tensor(scale_y, dtype=dtype, device=device)
    if s.ndim == 0:                            # scalar
        s = s.view(1, 1, 1)
    elif s.ndim == 1:
        if s.numel() == 3:                     # [3]
            s = s.view(1, 1, 3)
        elif s.numel() == Gmax:                # [G]
            s = s.view(1, Gmax, 1)
        else:
            raise ValueError(f"Incompatible 1D scale_y of length {s.numel()} for G={Gmax}")
    elif s.ndim == 2:
        r, c = s.shape
        if r == 1 and c in (1, 3):             # [1,1] or [1,3]
            s = s.view(1, 1, c)
        elif r == Gmax and c in (1, 3):        # [G,1] or [G,3]
            s = s.view(1, Gmax, c)
        else:
            raise ValueError(f"Incompatible 2D scale_y {tuple(s.shape)} for G={Gmax}")
    elif s.ndim == 3:
        b, g, k = s.shape
        if (b not in (1, B)) or (g not in (1, Gmax)) or (k not in (1, 3)):
            raise ValueError(f"Incompatible 3D scale_y {tuple(s.shape)} for B={B}, G={Gmax}")
    else:
        raise ValueError(f"scale_y must be 0–3D, got {s.ndim}D")

    # Expand to [B, Gmax, K]
    s = s.expand(B, Gmax, s.shape[-1])         # K = 1 or 3

    # --- Map per-graph scales to nodes: gather by batch_index ---
    idx = batch_idx.clamp_min(0).unsqueeze(-1).expand(B, N, s.shape[-1])  # [B,N,K]
    s_nodes = s.gather(dim=1, index=idx)        # [B,N,K]

    # If K==1, broadcast over 3 components
    if s_nodes.size(-1) == 1:
        s_nodes = s_nodes.expand(B, N, 3)

    out_pred = pred / s_nodes
    out_y    = y    / s_nodes

    return (out_pred.squeeze(0), out_y.squeeze(0)) if squeeze_back else (out_pred, out_y)

def _unscale_frame(pred_scaled: torch.Tensor,
                   y_scaled: torch.Tensor,
                   scale_y,
                   batch_index: torch.Tensor):
    """
    Unscale node-wise DOFs:
      - 2D frame: [u, v, rot]  -> D = 3
      - 3D frame: [u, v, w, thx, thy, thz] -> D = 6

    pred_scaled, y_scaled: [N,D] or [B,N,D]
    batch_index:           [N]   or [B,N]
    scale_y:
      - scalar         -> []
      - per-DOF        -> [D] or [1,D]
      - per-graph scal -> [G] or [G,1]
      - per-graph per-DOF -> [G,D]   (and optional batch versions [B,G,1/D])
    """
    assert pred_scaled.shape == y_scaled.shape
    device, dtype = pred_scaled.device, pred_scaled.dtype

    # Normalize to [B,N,D]
    if pred_scaled.ndim == 2:
        pred = pred_scaled.unsqueeze(0)
        y    = y_scaled.unsqueeze(0)
        batch_idx = batch_index.unsqueeze(0)
        squeeze_back = True
    else:
        pred = pred_scaled
        y    = y_scaled
        batch_idx = batch_index
        squeeze_back = False

    B, N, D = pred.shape
    if N == 0:
        return pred_scaled, y_scaled

    Gmax = int(batch_idx.max().item()) + 1

    # --- Normalize scale_y to [B', G', K] where K in {1, D} ---
    s = torch.as_tensor(scale_y, dtype=dtype, device=device)
    if s.ndim == 0:
        s = s.view(1, 1, 1)                      # scalar
    elif s.ndim == 1:
        if s.numel() == D:                       # [D] (your case if you skip view)
            s = s.view(1, 1, D)
        elif s.numel() == Gmax:                  # [G]
            s = s.view(1, Gmax, 1)
        else:
            raise ValueError(f"Incompatible 1D scale_y len={s.numel()} for G={Gmax}, D={D}")
    elif s.ndim == 2:
        r, c = s.shape
        if r == 1 and c in (1, D):               # [1,1] or [1,D] (your S_y shape)
            s = s.view(1, 1, c)
        elif r == Gmax and c in (1, D):          # [G,1] or [G,D]
            s = s.view(1, Gmax, c)
        else:
            raise ValueError(f"Incompatible 2D scale_y {tuple(s.shape)} for G={Gmax}, D={D}")
    elif s.ndim == 3:
        b, g, k = s.shape
        if (b not in (1, B)) or (g not in (1, Gmax)) or (k not in (1, D)):
            raise ValueError(f"Incompatible 3D scale_y {tuple(s.shape)} for B={B}, G={Gmax}, D={D}")
    else:
        raise ValueError(f"scale_y must be 0–3D, got {s.ndim}D")

    # Expand to [B, Gmax, K]
    s = s.expand(B, Gmax, s.shape[-1])           # K = 1 or D

    # Map per-graph scales to nodes
    idx = batch_idx.clamp_min(0).unsqueeze(-1).expand(B, N, s.shape[-1])  # [B,N,K]
    s_nodes = s.gather(dim=1, index=idx)         # [B,N,K]

    # If K == 1, broadcast to all DOFs
    if s_nodes.size(-1) == 1:
        s_nodes = s_nodes.expand(B, N, D)

    out_pred = pred / s_nodes
    out_y    = y    / s_nodes

    return (out_pred.squeeze(0), out_y.squeeze(0)) if squeeze_back else (out_pred, out_y)


def normalize_truss_coords(coord: np.ndarray, Lx_ref: float | None = None):
    """
    coord: (N, 2) array with columns [x, y]
    Lx_ref: reference length for normalization (if None, uses max(x)-min(x))
    Returns:
        coord_norm: (N, 2) normalized coordinates
        Lx_ref: the reference length actually used
    """
    coord = np.asarray(coord, dtype=float)
    x = coord[:, 0]
    y = coord[:, 1]

    # default reference = span in x
    if Lx_ref is None:
        Lx_ref = x.max() - x.min()

    if Lx_ref == 0:
        raise ValueError("Lx_ref is zero; cannot normalize with a zero reference length.")

    x_norm = (x - x.min()) / Lx_ref
    y_norm = (y - y.min()) / Lx_ref

    coord_norm = np.column_stack((x_norm, y_norm))
    return coord_norm, Lx_ref

def normalize_spacetruss_coords(coord: np.ndarray, Lx_ref: float | None = None):
    """
    coord: (N, 2) array with columns [x, y]
    Lx_ref: reference length for normalization (if None, uses max(x)-min(x))
    Returns:
        coord_norm: (N, 2) normalized coordinates
        Lx_ref: the reference length actually used
    """
    coord = np.asarray(coord, dtype=float)
    x = coord[:, 0]
    y = coord[:, 1]
    z = coord[:, 2]

    # default reference = span in x
    if Lx_ref is None:
        Lx_ref = x.max() - x.min()

    if Lx_ref == 0:
        raise ValueError("Lx_ref is zero; cannot normalize with a zero reference length.")

    x_norm = (x - x.min()) / Lx_ref
    y_norm = (y - y.min()) / Lx_ref
    z_norm = (z - z.min()) / Lx_ref

    coord_norm = np.column_stack((x_norm, y_norm, z_norm))
    return coord_norm, Lx_ref

def scale_planetruss_displacements(coord, A, F_char, E_mod, u, v, L=None, elem=None):
    """
    coord: (N, 2) node coordinates
    A:     (E, 1) or (E,) cross-section areas
    F_char: scalar
    E_mod:  scalar (Young's modulus)
    u, v:  (1, N) or (N,) nodal displacements in x and y
    L:     optional (E, 1) or (E,) element lengths; if None, provide elem
    elem:  optional (E, 2) integer node connectivity, used only if L is None

    Returns:
      u_scaled, v_scaled, S_y, A_char, L_char
    """
    coord = np.asarray(coord, dtype=float)
    A = np.asarray(A, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    # Make u, v shape (1, N) for clean broadcasting in the return
    if u.ndim == 1:
        u = u.reshape(1, -1)
    if v.ndim == 1:
        v = v.reshape(1, -1)

    # Element lengths
    if L is None:
        if elem is None:
            raise ValueError("Provide L or elem to compute element lengths.")
        elem = np.asarray(elem, dtype=int)
        p = coord[elem[:, 0], :]   # (E, 2)
        q = coord[elem[:, 1], :]   # (E, 2)
        L = np.linalg.norm(q - p, axis=1)
    else:
        L = np.asarray(L, dtype=float).reshape(-1)

    if L.size != A.size:
        raise ValueError(f"A and L must have same length. Got A={A.size}, L={L.size}.")

    A_char = A.mean()
    L_char = L.mean()

    if F_char == 0:
        raise ValueError("F_char is zero; cannot scale with zero force.")
    if L_char == 0:
        raise ValueError("Mean element length L_char is zero; check inputs.")

    S_y = (E_mod * A_char) / (F_char * L_char)

    u_scaled = u * S_y
    v_scaled = v * S_y

    return u_scaled, v_scaled, S_y, A_char, L_char


def scale_spacetruss_displacements(coord, A, F_char, E_mod, u, v, w, L=None, elem=None):
    """
    coord: (N, 2) node coordinates
    A:     (E, 1) or (E,) cross-section areas
    F_char: scalar
    E_mod:  scalar (Young's modulus)
    u, v:  (1, N) or (N,) nodal displacements in x and y
    L:     optional (E, 1) or (E,) element lengths; if None, provide elem
    elem:  optional (E, 2) integer node connectivity, used only if L is None

    Returns:
      u_scaled, v_scaled, S_y, A_char, L_char
    """
    coord = np.asarray(coord, dtype=float)
    A = np.asarray(A, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)

    # Make u, v, w shape (1, N) for clean broadcasting in the return
    if u.ndim == 1:
        u = u.reshape(1, -1)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    if w.ndim == 1:
        w = w.reshape(1, -1)

    # Element lengths
    if L is None:
        if elem is None:
            raise ValueError("Provide L or elem to compute element lengths.")
        elem = np.asarray(elem, dtype=int)
        p = coord[elem[:, 0], :]   # (E, 3)
        q = coord[elem[:, 1], :]   # (E, 3)
        L = np.linalg.norm(q - p, axis=1)
    else:
        L = np.asarray(L, dtype=float).reshape(-1)

    # if L.size != A.size:
    #     raise ValueError(f"A and L must have same length. Got A={A.size}, L={L.size}.")

    A_char = A.mean()
    L_char = L.mean()

    if F_char == 0:
        raise ValueError("F_char is zero; cannot scale with zero force.")
    if L_char == 0:
        raise ValueError("Mean element length L_char is zero; check inputs.")

    S_y = (E_mod * A_char) / (F_char * L_char)

    u_scaled = u * S_y
    v_scaled = v * S_y
    w_scaled = w * S_y

    return u_scaled, v_scaled, w_scaled, S_y, A_char, L_char


def normalize_parp(parp, coord_normalized, A_char, *, copy=True):
    """
    parp: (N, 3) array -> [x, y, A]
    coord_normalized: (N, 2) array -> normalized [x, y]
    A_char: scalar (mean cross-sectional area)
    copy: if True, return a new array; if False, modify parp in-place

    Returns:
        parp_norm: (N, 3) with first two cols replaced by coord_normalized,
                   third col divided by A_char
    """
    parp = np.asarray(parp, dtype=float)
    coord_normalized = np.asarray(coord_normalized, dtype=float)

    if parp.ndim != 2 or parp.shape[1] != 3:
        raise ValueError(f"'parp' must be (N,3); got {parp.shape}")
    if coord_normalized.ndim != 2 or coord_normalized.shape[1] != 2:
        raise ValueError(f"'coord_normalized' must be (N,2); got {coord_normalized.shape}")
    if parp.shape[0] != coord_normalized.shape[0]:
        raise ValueError("Row count mismatch between parp and coord_normalized.")
    if A_char == 0:
        raise ValueError("A_char is zero; cannot divide by zero.")

    out = parp.copy() if copy else parp
    out[:, :2] = coord_normalized
    out[:, 2] = out[:, 2] / A_char
    return out


def build_truss_parameters(parp, coord, Lx_ref, elem, A_char, copy=True):
    """
    Parameters
    ----------
    parp : ndarray of shape (E, 3)
        Input parameter array; its 3rd column (parp[:, 2]) contains
        the (unnormalized) cross-sectional areas A of the elements.
    coord : ndarray of shape (N, 2)
        Coordinates of nodes (x, y).
    Lx_ref : float
        Reference length for coordinate normalization.
    elem : ndarray of shape (E, 2)
        Element connectivity (node indices of each truss element).
        Assumed to be 0-based indices. If your indices are 1-based,
        use: elem = elem - 1 before calling this function.
    A_char : float
        Characteristic area for normalization of A.
    copy : bool, optional
        If True, work on a copy of `parp`; otherwise modify it in-place.

    Returns
    -------
    out : ndarray of shape (E, 3)
        For each element:
        [x_center_norm, y_center_norm, A_element_norm]
    """

    # normalize coordinates (whatever your normalize_truss_coords does)
    coord_norm, _ = normalize_truss_coords(coord, Lx_ref)

    # node indices for each element
    n1 = elem[:, 0]
    n2 = elem[:, 1]

    # coordinates of the two nodes of each element
    coord1 = coord_norm[n1, :]   # (E, 2)
    coord2 = coord_norm[n2, :]   # (E, 2)

    # center coordinates of each element
    center = 0.5 * (coord1 + coord2)  # (E, 2)

    # take A from the third column of *original* parp
    A_raw = parp[:, 2]
    A_norm = A_raw / A_char

    # build output
    out = parp.copy() if copy else parp
    out[:, :2] = center
    out[:, 2] = A_norm

    return out


def build_truss_parameters_edge(parp, elem, A_char, copy=True):
    """
    Parameters
    ----------
    parp : ndarray of shape (E, 3)
        Input parameter array; its 3rd column (parp[:, 2]) contains
        the (unnormalized) cross-sectional areas A of the elements.
    elem : ndarray of shape (E, 2)
        Element connectivity (node indices of each truss element).
        Assumed to be 0-based indices. If your indices are 1-based,
        use: elem = elem - 1 before calling this function.
    A_char : float
        Characteristic area for normalization of A.
    copy : bool, optional
        If True, work on a copy of `parp`; otherwise modify it in-place.

    Returns
    -------
    out : ndarray of shape (E, 3)
        For each element:
        [elem, A_element_norm]    """


    # take A from the third column of *original* parp
    A_raw = parp[:, 2]
    A_norm = A_raw / A_char

    # build output
    out = parp.copy() if copy else parp
    out[:, :2] = elem
    out[:, 2] = A_norm

    return out


def make_or_update_parp(coord_normalized, loads, F_char, parp=None, nodes=(2, 5), one_based=True):
    """
    coord_normalized : (N, 2) normalized [x, y] for each node
    loads            : (N, 2) [Fx, Fy] for each node
    F_char           : scalar, characteristic vertical load
    parp             : optional existing array to update; if provided and has shape (k, 3),
                       its first two columns are replaced by coords and the third by Fy/F_char.
    nodes            : iterable of node IDs to include (default: nodes 2 and 5)
    one_based        : True if 'nodes' are MATLAB-style (1-based). If False, treat as 0-based.

    Returns:
        parp_out : (k, 3) with rows per node in 'nodes': [x_norm, y_norm, Fy_norm]
                   where Fy_norm = Fy / F_char
    """
    if F_char == 0:
        raise ValueError("F_char is zero; cannot normalize loads.")

    # indices for selection
    # idx = np.array(nodes, dtype=int)
    idx = np.atleast_1d(nodes).astype(int)
    if one_based:
        idx = idx - 1  # convert MATLAB 1-based to Python 0-based

    # grab coords and vertical loads
    xy = np.asarray(coord_normalized, dtype=float)[idx, :]        # (k, 2)
    Fy = np.asarray(loads, dtype=float)[idx, 1]                   # (k,)

    # normalized vertical loads
    Fy_norm = Fy / float(F_char)

    if parp is not None:
        parp_out = np.asarray(parp, dtype=float).copy()
        if parp_out.shape != (len(idx), 3):
            raise ValueError(f"parp has shape {parp_out.shape}, expected ({len(idx)}, 3).")
        parp_out[:, :2] = xy
        parp_out[:, 2]  = Fy_norm
    else:
        parp_out = np.column_stack([xy, Fy_norm])                 # (k, 3)

    return parp_out

def make_or_update_parp3(coord_normalized, loads, F_char, parp=None, nodes=(2, 5), one_based=True):
    """
    coord_normalized : (N, 3) normalized [x, y, z] for each node
    loads            : (N, 3) [Fx, Fy, Fz] for each node
    F_char           : scalar, characteristic vertical load
    parp             : optional existing array to update; if provided and has shape (k, 4),
                       its first two columns are replaced by coords and the third by Fy/F_char.
    nodes            : iterable of node IDs to include (default: nodes 2 and 5)
    one_based        : True if 'nodes' are MATLAB-style (1-based). If False, treat as 0-based.

    Returns:
        parp_out : (k, 4) with rows per node in 'nodes': [x_norm, y_norm, z_norm, Fz_norm]
                   where Fz_norm = Fz / F_char
    """
    if F_char == 0:
        raise ValueError("F_char is zero; cannot normalize loads.")

    # indices for selection
    # idx = np.array(nodes, dtype=int)
    idx = np.atleast_1d(nodes).astype(int)
    if one_based:
        idx = idx - 1  # convert MATLAB 1-based to Python 0-based

    # grab coords and vertical loads
    xyz = np.asarray(coord_normalized, dtype=float)[idx, :]        # (k, 3)
    Fz = np.asarray(loads, dtype=float)[idx, 2]                   # (k,)

    # normalized vertical loads
    Fz_norm = Fz / float(F_char)

    if parp is not None:
        parp_out = np.asarray(parp, dtype=float).copy()
        if parp_out.shape != (len(idx), 4):
            raise ValueError(f"parp has shape {parp_out.shape}, expected ({len(idx)}, 4).")
        parp_out[:, :3] = xyz
        parp_out[:, 3]  = Fz_norm
    else:
        parp_out = np.column_stack([xyz, Fz_norm])                 # (k, 4)

    return parp_out

def scale_planeframe_displacements(coord, A, F_char, E, I, u, v, theta, L=None, elem=None):
    """
    coord: (N, 2) node coordinates
    A:     (E, 1) or (E,) cross-section areas
    F_char: scalar
    E:  scalar (Young's modulus)
    I: scaler (Moment of inertia)
    u, v, theta:  (1, N) or (N,) nodal displacements in x and y and rotation
    L:     optional (E, 1) or (E,) element lengths; if None, provide elem
    elem:  optional (E, 2) integer node connectivity, used only if L is None

    Returns:
      u_scaled, v_scaled, theta_scaled, S_y, A_char, L_char
    """
    coord = np.asarray(coord, dtype=float)
    A = np.asarray(A, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    theta = np.asarray(theta, dtype=float)

    # Make u, v shape (1, N) for clean broadcasting in the return
    if u.ndim == 1:
        u = u.reshape(1, -1)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    if theta.ndim == 1:
        theta = theta.reshape(1, -1)

    # Element lengths
    if L is None:
        if elem is None:
            raise ValueError("Provide L or elem to compute element lengths.")
        elem = np.asarray(elem, dtype=int)
        p = coord[elem[:, 0], :]   # (E, 2)
        q = coord[elem[:, 1], :]   # (E, 2)
        L = np.linalg.norm(q - p, axis=1)
    else:
        L = np.asarray(L, dtype=float).reshape(-1)

    # if L.size != A.size:
    #     raise ValueError(f"A and L must have same length. Got A={A.size}, L={L.size}.")

    A_char = A.mean()
    L_char = L.mean()

    if F_char == 0:
        raise ValueError("F_char is zero; cannot scale with zero force.")
    if L_char == 0:
        raise ValueError("Mean element length L_char is zero; check inputs.")

    s_u = (E * A_char) / (F_char * L_char)  # for axial displacement u
    s_v = (E * I) / (F_char * (L_char ** 3))  # for transverse displacement v
    s_th = (E * I) / (F_char * (L_char ** 2))  # for rotation theta
    # Stack into (3,) or (N,3) depending on inputs; reshape for broadcasting if needed
    # S_y = np.array([s_u, s_v, s_th], dtype=float).reshape(1, 3)
    #
    # u_scaled = u * s_u
    # v_scaled = v * s_v
    # theta_scaled = theta * s_th

    s_th_eff = s_th / L_char
    seq = (s_u * s_v * s_th_eff) ** (1.0 / 3.0)

    S_y = np.array([seq, seq, seq * L_char], dtype=float).reshape(1, 3)

    u_scaled = u * S_y[0, 0]
    v_scaled = v * S_y[0, 1]
    theta_scaled = theta * S_y[0, 2]

    return u_scaled, v_scaled, theta_scaled, S_y, A_char, L_char


def scale_spaceframe_displacements(coord, A, Iy, Iz, J, F_char, E, G, u, v, w, thx, thy, thz, L=None, elem=None):
    """
    coord: (N, 2) node coordinates
    A:     (E, 1) or (E,) cross-section areas
    F_char: scalar
    E:  scalar (Young's modulus)
    I: scaler (Moment of inertia)
    u, v, theta:  (1, N) or (N,) nodal displacements in x and y and rotation
    L:     optional (E, 1) or (E,) element lengths; if None, provide elem
    elem:  optional (E, 2) integer node connectivity, used only if L is None

    Returns:
      u_scaled, v_scaled, theta_scaled, S_y, A_char, L_char
    """
    coord = np.asarray(coord, dtype=float)
    A = np.asarray(A, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    thx = np.asarray(thx, dtype=float)
    thy = np.asarray(thy, dtype=float)
    thz = np.asarray(thz, dtype=float)

    # Make u, v, w, thx, thy, thz shape (1, N) for clean broadcasting in the return
    if u.ndim == 1:
        u = u.reshape(1, -1)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    if thx.ndim == 1:
        thx = thx.reshape(1, -1)
    if thy.ndim == 1:
        thy = thy.reshape(1, -1)
    if thz.ndim == 1:
        thz = thz.reshape(1, -1)

    # Element lengths
    if L is None:
        if elem is None:
            raise ValueError("Provide L or elem to compute element lengths.")
        elem = np.asarray(elem, dtype=int)
        p = coord[elem[:, 0], :]   # (E, 3)
        q = coord[elem[:, 1], :]   # (E, 3)
        L = np.linalg.norm(q - p, axis=1)
    else:
        L = np.asarray(L, dtype=float).reshape(-1)

    # if L.size != A.size:
    #     raise ValueError(f"A and L must have same length. Got A={A.size}, L={L.size}.")

    A_char = A.mean()
    Iy_char = Iy.mean()
    Iz_char = Iz.mean()
    J_char = J.mean()
    L_char = L.mean()

    if F_char == 0:
        raise ValueError("F_char is zero; cannot scale with zero force.")
    if L_char == 0:
        raise ValueError("Mean element length L_char is zero; check inputs.")

    s_u = (E * A_char) / (F_char * L_char)  # for axial displacement u
    s_v = (E * Iz_char) / (F_char * (L_char ** 3))  # transverse v  (bending about z)
    s_w = (E * Iy_char) / (F_char * (L_char ** 3))  # transverse w  (bending about y)
    s_thy = (E * Iy_char) / (F_char * (L_char ** 2))  # rotation about y → same Iy as w
    s_thz = (E * Iz_char) / (F_char * (L_char ** 2))  # for rotation theta
    s_thx = (G * J_char) / (F_char * (L_char ** 2))  # for rotation theta
    # Stack into (3,) or (N,3) depending on inputs; reshape for broadcasting if needed
    # S_y = np.array([s_u, s_v, s_w, s_thx, s_thy, s_thz], dtype=float).reshape(1, 6)
    #
    # u_scaled = u * s_u
    # v_scaled = v * s_v
    # w_scaled = w * s_w
    # thx_scaled = thx * s_thx
    # thy_scaled = thy * s_thy
    # thz_scaled = thz * s_thz

    s_thx_eff = s_thx / L_char
    s_thy_eff = s_thy / L_char
    s_thz_eff = s_thz / L_char

    seq = (s_u * s_v * s_w * s_thx_eff * s_thy_eff * s_thz_eff) ** (1.0 / 6.0)

    S_y = np.array([seq, seq, seq, seq * L_char, seq * L_char, seq * L_char], dtype=float).reshape(1, 6)

    u_scaled = u * S_y[0, 0]
    v_scaled = v * S_y[0, 1]
    w_scaled = w * S_y[0, 2]
    thx_scaled = thx * S_y[0, 3]
    thy_scaled = thy * S_y[0, 4]
    thz_scaled = thz * S_y[0, 5]

    return u_scaled, v_scaled, w_scaled, thx_scaled, thy_scaled, thz_scaled, \
        S_y, A_char, Iy_char, Iz_char, J_char, L_char

def scale_spaceframe_displacements_mix(coord, A, Iy, Iz, J, F_char, E, G, u, v, w, thx, thy, thz, L=None, elem=None):
    """
    coord: (N, 2) node coordinates
    A:     (E, 1) or (E,) cross-section areas
    F_char: scalar
    E:  scalar (Young's modulus)
    I: scaler (Moment of inertia)
    u, v, theta:  (1, N) or (N,) nodal displacements in x and y and rotation
    L:     optional (E, 1) or (E,) element lengths; if None, provide elem
    elem:  optional (E, 2) integer node connectivity, used only if L is None

    Returns:
      u_scaled, v_scaled, theta_scaled, S_y, A_char, L_char
    """
    coord = np.asarray(coord, dtype=float)
    A = np.asarray(A, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    thx = np.asarray(thx, dtype=float)
    thy = np.asarray(thy, dtype=float)
    thz = np.asarray(thz, dtype=float)

    # Make u, v, w, thx, thy, thz shape (1, N) for clean broadcasting in the return
    if u.ndim == 1:
        u = u.reshape(1, -1)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    if thx.ndim == 1:
        thx = thx.reshape(1, -1)
    if thy.ndim == 1:
        thy = thy.reshape(1, -1)
    if thz.ndim == 1:
        thz = thz.reshape(1, -1)

    # Element lengths
    if L is None:
        if elem is None:
            raise ValueError("Provide L or elem to compute element lengths.")
        elem = np.asarray(elem, dtype=int)
        p = coord[elem[:, 0], :]   # (E, 3)
        q = coord[elem[:, 1], :]   # (E, 3)
        L = np.linalg.norm(q - p, axis=1)
    else:
        L = np.asarray(L, dtype=float).reshape(-1)

    # if L.size != A.size:
    #     raise ValueError(f"A and L must have same length. Got A={A.size}, L={L.size}.")

    A_char = A.mean()
    Iy_char = Iy.mean()
    Iz_char = Iz.mean()
    J_char = J.mean()
    L_char = L.mean()

    if F_char == 0:
        raise ValueError("F_char is zero; cannot scale with zero force.")
    if L_char == 0:
        raise ValueError("Mean element length L_char is zero; check inputs.")

    # kax = (E * A_char) / L_char  # axial stiffness
    # kbz = (12 * E * Iz_char) / L_char ** 3  # bending about z
    # kby = (12 * E * Iy_char) / L_char ** 3  # bending about y
    # kt = (G * J_char) / L_char  # torsion stiffness
    #
    # keq = np.sqrt(kax ** 2 + kbz ** 2 + kby ** 2 + kt ** 2)
    #
    # S_y = keq / F_char  # dimensionless scale

    S_y = 1.0

    u_scaled = u * S_y
    v_scaled = v * S_y
    w_scaled = w * S_y
    thx_scaled = thx * S_y
    thy_scaled = thy * S_y
    thz_scaled = thz * S_y

    return u_scaled, v_scaled, w_scaled, thx_scaled, thy_scaled, thz_scaled, \
        S_y, A_char, Iy_char, Iz_char, J_char, L_char


def save_epoch_attention_mat(args, epoch, epoch_attn, save_dir):
    """
    epoch_attn: list of dicts (one per mini-batch)
        each dict has keys: scores, alpha, batch (torch tensors on CPU)
    """
    os.makedirs(save_dir, exist_ok=True)

    mat_data = {}
    for i, attn in enumerate(epoch_attn):
        mat_data[f"alpha_batch_{i}"]  = attn["alpha"].numpy()
        mat_data[f"batch_batch_{i}"]  = (
            attn["batch"].numpy() if attn["batch"] is not None else None
        )

    mat_data["num_batches"] = len(epoch_attn)

    filename = os.path.join(save_dir, f"epoch_{epoch:04d}_last_attention_{args.data}_{args.model}.mat")
    savemat(filename, mat_data)

