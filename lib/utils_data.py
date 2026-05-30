import scipy.io as sio
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.loader import DataLoader
from .utils_graph import edge_lengths, normalize_truss_coords, normalize_spacetruss_coords, \
      scale_planetruss_displacements, scale_spaceframe_displacements, \
      scale_spacetruss_displacements, \
      make_or_update_parp, make_or_update_parp3, \
      scale_planeframe_displacements
from scipy.io import savemat
import mat73
from types import SimpleNamespace


def _canonicalize_Sy(S_y, d):
    """
        Canonicalize an input vector v to shape (1, d).

        Rules:
        - If v is a scalar → repeat to length d.
        - If v is array-like with size 1 → repeat to length d.
        - If v is array-like with size d → reshape to (1, d).
        - Otherwise → raise ValueError.
        """
    S_y = np.asarray(S_y, dtype=np.float64)

    # Scalar
    if S_y.ndim == 0:
        return np.full((1, d), float(S_y), dtype=np.float64)

    # Only one value in array
    if S_y.size == 1:
        val = float(S_y.reshape(-1)[0])
        return np.full((1, d), val, dtype=np.float64)

    # Match expected dimension
    flat = S_y.reshape(-1)
    if flat.size == d:
        return flat.reshape(1, d)

    raise ValueError(
        f"Unexpected vector size {S_y.shape}; expected scalar, 1-element, or length {d}."
    )


def _as_cell(lst):
    """Convert a Python list of arrays into a NumPy object array (MATLAB cell)."""
    cell = np.empty(len(lst), dtype=object)
    for i, v in enumerate(lst):
        cell[i] = v
    return cell


def generate_planeframe_data_loader_var_load(args, config):

    # load the data
    mat = sio.loadmat(config['data']['datapath'])
    num_train_input = config['train']['num_train_input']
    u = np.stack(mat['u'][0])  # list of M elements
    v = np.stack(mat['v'][0])  # list of M elements
    theta = np.stack(mat['theta'][0])  # list of M elements
    coors = np.stack(mat['coors'][0])    # list of M elements
    par = np.stack(mat['input_var'][0])    # list of M elements
    ic_flag = np.stack(mat['total_ic_flag'][0])   # list of M elements

    # ------- globals (v7) -------
    globals_raw = {
        "A": load_global_field(mat, "A", stack=False),
        "I": load_global_field(mat, "I", stack=False),
        "E": load_global_field(mat, "E", stack=False),
        "Lx_ref": load_global_field(mat, "Lx", stack=False),
        "elems": load_global_field(mat, "conn", stack=False),
    }

    # ------------- check + convert globals to NumPy -------------
    required_names = ["A", "E", "I", "Lx_ref", "elems"]

    for name in required_names:
        if globals_raw.get(name) is None:
            raise ValueError(f"Global '{name}' could not be loaded from MAT file.")
        globals_raw[name] = np.asarray(globals_raw[name])

    # unpack if you still like explicit names
    A = globals_raw["A"]
    E = globals_raw["E"]
    I = globals_raw["I"]
    Lx_ref = globals_raw["Lx_ref"]
    elems_global = globals_raw["elems"]
    elems_np = np.asarray(elems_global, dtype=np.int64)  # (M, 2), 1-based
    elems = elems_np - 1

    F = np.stack(mat['loads'][0])

    '''
    prepare the data to support batchwise training
    '''
    # find the maximum number of nodes
    datasize = len(u)
    max_pde_nodes = 0
    max_par_nodes = 0
    max_bc_nodes = 0
    for i in range(datasize):
        num_pde = np.sum(1-ic_flag[i])
        if num_pde > max_pde_nodes:
            max_pde_nodes = num_pde
        num_par_ = par[i].shape[0]
        if num_par_ > max_par_nodes:
            max_par_nodes = num_par_
        num_bc = np.sum(ic_flag[i])
        if num_bc > max_bc_nodes:
            max_bc_nodes = num_bc
    max_pde_nodes = int(max_pde_nodes)
    max_bc_nodes = int(max_bc_nodes)
    max_par_nodes = int(max_par_nodes)

    # --- keep track of per-sample info only for TEST range
    test_orig_coors = []
    test_orig_u = []
    test_orig_v = []
    test_orig_theta = []
    test_orig_par = []
    test_orig_A = []
    test_orig_loads = []

    test_norm_coors_s = []
    test_norm_u = []
    test_norm_v = []
    test_norm_theta = []
    test_norm_parp = []
    test_flag = []

    test_meta = {
        "pde_idx": [], "bc_idx": [],
        "F_char": [], "A_char": [], "L_char": [], "S_y": []
        # store whatever normalize_truss_coords returns as 2nd item
    }

    # split the data
    # bar1 = [0,int(0.7*datasize)]
    # bar2 = [int(0.7*datasize),int(0.8*datasize)]
    # bar3 = [int(0.8*datasize),int(datasize)]
    bar1 = [0, num_train_input]
    bar2 = [num_train_input, num_train_input + 2000]
    bar3 = [num_train_input + 2000, num_train_input + 4000]

    # Precompute the integer range for quick membership test
    test_indices = set(range(bar3[0], min(bar3[1], datasize)))

    # append zeros to the data
    uT = []
    vT = []
    thetaT = []
    coorT = []
    parT = []
    par_flagT = []
    flagT = []
    S_yT = []
    for i in range(datasize):
        # extract the index of pde nodes and bc nodes
        pde_idx = np.where(ic_flag[i]==0)[1]
        bc_idx = np.where(ic_flag[i]==1)[1]
        num_pde = np.size(pde_idx)
        num_bc = np.size(bc_idx)
        # re-organize coors
        coorp = coors[i]
        coorp_norm, _ = normalize_truss_coords(coorp, Lx_ref)

        coorp_s = np.concatenate((coorp_norm[pde_idx, :], np.zeros((max_pde_nodes - num_pde, 2)), coorp_norm[bc_idx, :],
                                  np.zeros((max_bc_nodes - num_bc, 2))), 0)  # (max_pde+max_bc,2)
        coorp_s = np.expand_dims(coorp_s, 0)  # (1,max_pde+max_bc,2)
        coorT.append(coorp_s)

        # re-organize solution
        up = u[i]
        vp = v[i]
        thetap = theta[i]
        Ap = A
        loads = F[i]
        Fy = loads[:, 1]  # extract y-direction loads
        F_char = np.max(np.abs(Fy))  # take maximum magnitude of vertical load
        up_s, vp_s, thetap_s, S_y, A_char, L_char = scale_planeframe_displacements(coorp, Ap, F_char, E, I,
                                                                                   up, vp, thetap, elem=elems)

        S_y = _canonicalize_Sy(S_y, loads.shape[1])  # (1,3)
        S_yT.append(S_y)

        up_s = np.concatenate((up_s[:, pde_idx], np.zeros((1, max_pde_nodes - num_pde)), up_s[:, bc_idx],
                               np.zeros((1, max_bc_nodes - num_bc))), -1)  # (1, max_pde+max_bc)
        uT.append(up_s)

        vp_s = np.concatenate((vp_s[:, pde_idx], np.zeros((1, max_pde_nodes - num_pde)), vp_s[:, bc_idx],
                               np.zeros((1, max_bc_nodes - num_bc))), -1)  # (1, max_pde+max_bc)
        vT.append(vp_s)
        thetap_s = np.concatenate((thetap_s[:, pde_idx], np.zeros((1, max_pde_nodes - num_pde)), thetap_s[:, bc_idx],
                                 np.zeros((1, max_bc_nodes - num_bc))), -1)  # (1, max_pde+max_bc)
        thetaT.append(thetap_s)

        # re-organize parameters
        parpv = par[i]
        load_nodes = tuple(np.where(Fy!=0)[0])
        parpv_norm = make_or_update_parp(coorp_norm, loads, 1.0, parp=parpv, nodes=load_nodes, one_based=False)
        num_par = parpv_norm.shape[0]
        parp_s = np.concatenate((parpv_norm, np.zeros((max_par_nodes-num_par,3))), 0)    # (max_par,3)
        par_flag = np.concatenate((np.ones_like(parpv), np.zeros((max_par_nodes-num_par,3))), 0)    # (max_par,3)
        parp_s = np.expand_dims(parp_s, 0)    # (1,max_par,3)
        par_flag = np.expand_dims(par_flag, 0)    # (1,max_par,3)
        parT.append(parp_s)
        par_flagT.append(par_flag)
        # re-organize ic flag
        flagp = ic_flag[i]
        flagp = np.concatenate((flagp[:,pde_idx], -np.ones((1,max_pde_nodes-num_pde)), flagp[:,bc_idx],
                                -np.ones((1,max_bc_nodes-num_bc))), -1)    # (1, max_pde+max_bc)
        flagT.append(flagp)

        if i in test_indices:
            # originals (no padding, original ordering)
            test_orig_coors.append(coorp)
            test_orig_u.append(up)
            test_orig_v.append(vp)
            test_orig_theta.append(thetap)
            test_orig_par.append(parpv)
            test_orig_A.append(Ap)
            test_orig_loads.append(loads)

            # normalized + padded (the exact tensors fed to the model at test time)
            test_norm_coors_s.append(coorp_s[0])  # drop leading dim
            test_norm_u.append(up_s[0])
            test_norm_v.append(vp_s[0])
            test_norm_theta.append(thetap_s[0])
            test_norm_parp.append(parp_s[0])
            test_flag.append(flagp[0])

            # metadata for inverse-transform / reconstruction
            test_meta["pde_idx"].append(pde_idx.astype(np.int32))
            test_meta["bc_idx"].append(bc_idx.astype(np.int32))
            test_meta["F_char"].append(np.float64(F_char))
            test_meta["A_char"].append(np.asarray(A_char, dtype=np.float64))
            test_meta["L_char"].append(np.asarray(L_char, dtype=np.float64))
            test_meta["S_y"].append(np.asarray(S_y, dtype=np.float64))

    uT = np.concatenate(tuple(uT), 0)    # (M, max_node)
    vT = np.concatenate(tuple(vT), 0)  # (M, max_node)
    thetaT = np.concatenate(tuple(thetaT), 0)  # (M, max_node)
    coorT = np.concatenate(tuple(coorT), 0)    # (M, max_node, 2)
    parT = np.concatenate(tuple(parT), 0)    # (M, max_par_nodes,3)
    flagT = np.concatenate(tuple(flagT), 0)    # (M, max_node)
    par_flagT = np.concatenate(tuple(par_flagT), 0)[:,:,0]    # (M, max_par_nodes)
    S_yT = np.concatenate(tuple(S_yT), 0)  # (M, 3)
    uT = torch.from_numpy(uT)
    vT = torch.from_numpy(vT)
    thetaT = torch.from_numpy(thetaT)
    coorT = torch.from_numpy(coorT)
    parT = torch.from_numpy(parT)
    flagT = torch.from_numpy(flagT)
    par_flagT = torch.from_numpy(par_flagT)
    S_yT = torch.from_numpy(S_yT)


    train_dataset = torch.utils.data.TensorDataset(parT[bar1[0]:bar1[1],:,:],
            coorT[bar1[0]:bar1[1],:], uT[bar1[0]:bar1[1],:], vT[bar1[0]:bar1[1],:], thetaT[bar1[0]:bar1[1],:],
            flagT[bar1[0]:bar1[1],:], par_flagT[bar1[0]:bar1[1],:], S_yT[bar1[0]:bar1[1],:])
    val_dataset = torch.utils.data.TensorDataset(parT[bar2[0]:bar2[1],:,:],
            coorT[bar2[0]:bar2[1],:], uT[bar2[0]:bar2[1],:], vT[bar2[0]:bar2[1],:], thetaT[bar2[0]:bar2[1],:],
            flagT[bar2[0]:bar2[1],:], par_flagT[bar2[0]:bar2[1],:], S_yT[bar2[0]:bar2[1],:])
    test_dataset = torch.utils.data.TensorDataset(parT[bar3[0]:bar3[1],:,:],
            coorT[bar3[0]:bar3[1],:], uT[bar3[0]:bar3[1],:], vT[bar3[0]:bar3[1],:], thetaT[bar3[0]:bar3[1],:],
            flagT[bar3[0]:bar3[1],:], par_flagT[bar3[0]:bar3[1],:], S_yT[bar3[0]:bar3[1],:])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batchsize'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['train']['batchsize'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # NOTE: the lists below were populated in your loop only for test indices:
    #   test_orig_coors, test_orig_u, test_orig_v, test_orig_par,
    #   test_norm_coors_s, test_norm_u_s, test_norm_v_s, test_norm_parp_s, test_norm_flag_s,
    #   test_meta[...]  (pde_idx, bc_idx, F_char, A_char, L_char, S_y, coord_norm_aux)

    mat_payload = {
        # Originals (ragged -> MATLAB cell arrays)
        "orig_coors": _as_cell(test_orig_coors),  # each cell: (Ni, 2)
        "orig_u": _as_cell(test_orig_u),  # each cell: (1, Ni)
        "orig_v": _as_cell(test_orig_v),  # each cell: (1, Ni)
        "orig_theta": _as_cell(test_orig_theta),  # each cell: (1, Ni)
        "orig_par": _as_cell(test_orig_par),  # each cell: (Pi, 3)
        "orig_A": _as_cell(test_orig_A),
        "orig_loads": _as_cell(test_orig_loads),

        # Normalized & padded (uniform -> numeric arrays)
        # Shapes in MATLAB: size(...)= [Nt, max_node, 2], etc.
        "norm_coors": np.stack(test_norm_coors_s, axis=0).astype(np.float64),
        "norm_u": np.stack(test_norm_u, axis=0).astype(np.float64),
        "norm_v": np.stack(test_norm_v, axis=0).astype(np.float64),
        "norm_theta": np.stack(test_norm_theta, axis=0).astype(np.float64),
        "norm_parp": np.stack(test_norm_parp, axis=0).astype(np.float64),
        "flag": np.stack(test_flag, axis=0).astype(np.float64),

        # Metadata (ragged -> cell arrays; scalars -> numeric)
        "pde_idx": _as_cell([pi.astype(np.int32) for pi in test_meta["pde_idx"]]),
        "bc_idx": _as_cell([bi.astype(np.int32) for bi in test_meta["bc_idx"]]),
        "F_char": np.asarray(test_meta["F_char"], dtype=np.float64),  # (Nt,)

        # These may be scalars or arrays per sample -> store as cells
        "A_char": _as_cell(test_meta["A_char"]),
        "L_char": _as_cell(test_meta["L_char"]),
        "S_y": _as_cell(test_meta["S_y"]),

        # Helpful globals
        "max_pde_nodes": np.int32(max_pde_nodes),
        "max_bc_nodes": np.int32(max_bc_nodes),
        "max_par_nodes": np.int32(max_par_nodes),
        "test_start": np.int32(bar3[0]),
        "test_end": np.int32(min(bar3[1], datasize)),
        "E": np.asarray(E, dtype=np.float64),
        "I": np.asarray(I, dtype=np.float64),
        "elem": elems.astype(np.int32),
    }

    savemat(r'./res/saved_models/test_cache_{}_{}.mat'.format(args.data, args.model), mat_payload, do_compression=True)

    # store the number of nodes of different types
    num_nodes_list = (max_pde_nodes, max_bc_nodes, max_par_nodes)

    return train_loader, val_loader, test_loader, num_nodes_list


def generate_graphplaneframe_data_loader(args, config):
    # ------------------ splits ------------------
    n_train = int(config['train']['num_train'])
    n_val   = int(config['train']['num_val'])
    n_test  = int(config['train']['num_test'])
    total_num_data = n_train + n_val + n_test

    # ------------------ load .mat ------------------
    mat = sio.loadmat(config['data']['datapath'], squeeze_me=True, struct_as_record=False)
    graphs_mat = mat['graphs']

    # array of structs -> list
    if isinstance(graphs_mat, np.ndarray):
        graphs = list(graphs_mat.ravel())
    else:
        graphs = [graphs_mat]

    # ------- globals (v7) -------
    globals_raw = {
        "A": load_global_field(mat, "A", stack=False),
        "I": load_global_field(mat, "I", stack=False),
        "E": load_global_field(mat, "E", stack=False),
        "Lx_ref": load_global_field(mat, "Lx", stack=False),
        "elems": load_global_field(mat, "elements", stack=False),
        "bc": load_global_field(mat, "bc", stack=False),
    }

    # ------------- check + convert globals to NumPy -------------
    required_names = ["A", "E", "I", "Lx_ref", "elems", "bc"]

    for name in required_names:
        if globals_raw.get(name) is None:
            raise ValueError(f"Global '{name}' could not be loaded from MAT file.")
        globals_raw[name] = np.asarray(globals_raw[name])

    # unpack if you still like explicit names
    A_global = globals_raw["A"]
    E_global = globals_raw["E"]
    I_global = globals_raw["I"]
    Lx_ref_global = globals_raw["Lx_ref"]
    elems_global = globals_raw["elems"]
    bc_global = globals_raw["bc"]
    bc_global = bc_global.astype(np.int64)

    # ------------------ clip to requested total_num_data ------------------
    n_available = len(graphs)
    if total_num_data > n_available:
        print(f"[generate_graphplaneframe_data_loader] "
              f"Warning: requested {total_num_data} samples, "
              f"but MAT file only has {n_available}; truncating.")
        total_num_data = n_available

    graphs = graphs[:total_num_data]

    data_list = []
    for g in graphs:
        # -------- read fields (NumPy first) --------
        pos_np   = np.asarray(g.nodes)                 # (N, d)
        elems_np = np.asarray(elems_global, dtype=np.int64)  # (M, 2), 1-based
        loads_np = np.asarray(getattr(g, 'loads', np.zeros_like(pos_np)))
        bc_mask_np = np.asarray(bc_global)

        target   = getattr(g, 'target', None)
        if target is None:
            raise ValueError("Missing 'target' in a case; Option A expects 'target' per graph.")
        y = torch.tensor(np.asarray(target), dtype=torch.float32)  # (N,d) or (M,1) or scalar

        # -------- torch tensors --------
        pos = torch.tensor(pos_np, dtype=torch.float32)            # (N,d)
        A_phys = torch.tensor(A_global, dtype=torch.float32)
        N, d = pos.shape

        # Edge index (0-based), then make undirected
        edge_index_phys = torch.from_numpy(elems_np - 1).t().contiguous()  # (2, M)

        # -------- E from global E (scalar or array -> scalar for scaling) --------
        A_char = scalar_from_global(A_global, name="A")
        E_char = scalar_from_global(E_global, name="E")
        I_char = scalar_from_global(I_global, name="I")
        Lx_ref_val = scalar_from_global(Lx_ref_global, name="Lx_ref")

        # to_undirected and duplicate A accordingly
        edge_index_gnn = to_undirected(
            edge_index_phys, num_nodes=N
        )

        # -------- geometry-derived edge features --------
        L, rel = edge_lengths(pos, edge_index_gnn)
        cs = rel / L.clamp_min(1e-9)  # [E,2] = (cosx, cosy)

        # typical (physical) length from *physical* edges
        L_phys, _ = edge_lengths(pos, edge_index_phys)
        L_char = L_phys.mean().clamp_min(1e-12)  # scalar, guard tiny

        # -------- normalize node coords --------
        # simple left-bottom anchoring and global scale

        x_left = pos[:, 0].min()
        y_bottom = pos[:, 1].min()
        pos_norm = torch.stack([(pos[:, 0] - x_left) / Lx_ref_val,
                                (pos[:, 1] - y_bottom) / Lx_ref_val], dim=1)

        # -------- loads: per-graph characteristic force for scaling --------
        loads = torch.tensor(loads_np, dtype=torch.float32)   # (N,d)
        Fy = loads[:, 1] if loads.shape[1] >= 2 else torch.zeros(N, dtype=torch.float32)
        F_char = Fy.abs().max()
        if not torch.isfinite(F_char) or F_char.item() == 0.0:
            F_char = torch.tensor(1.0, dtype=torch.float32)  # avoid divide-by-zero

        # node features (example): normalized coords + scaled vertical load
        Fy_scaled = (Fy / F_char).unsqueeze(1)  # (N,1)
        x_node = torch.cat([pos_norm, Fy.unsqueeze(1)], dim=1).float()  # (N,3)

        # choose your edge_attr; here: just direction cosines
        edge_attr_gnn = torch.cat([cs, L/Lx_ref_val], dim=1).float()

        # -------- target scaling (node-level assumed) --------

        s_u = (E_char * A_char) / (F_char * L_char)  # for axial displacement u
        s_v = (E_char * I_char) / (F_char * (L_char ** 3))  # for transverse displacement v
        s_th = (E_char * I_char) / (F_char * (L_char ** 2))  # for rotation theta

        # Stack into (3,) or (N,3) depending on inputs; reshape for broadcasting if needed
        # S_y = torch.stack([s_u, s_v, s_th], dim=-1)
        # S_y = S_y.to(y.dtype).to(y.device)
        # if S_y.ndim == 1:  # scalar params -> shape (3,)
        #     S_y = S_y.view(1, 3)

        s_th_eff = s_th / L_char
        seq = (s_u * s_v * s_th_eff) ** (1.0 / 3.0)

        S_y = torch.stack([seq, seq, seq * L_char], dim=-1)
        S_y = S_y.to(y.dtype).to(y.device)
        if S_y.ndim == 1:  # scalar params -> shape (3,)
            S_y = S_y.view(1, 3)

        y_scaled = y * S_y  # broadcasts over rows

        # -------- build Data --------
        data = Data(
            x=x_node,                 # or x_node if you want loads too: x=x_node
            pos=pos_norm,               # positions used by message passing (normalized)
            pos_raw=pos,                # keep raw coords if your layers need them
            edge_index=edge_index_gnn,  # undirected
            edge_attr=edge_attr_gnn,    # [cosx, cosy]
            y=y_scaled,                 # scaled targets
        )

        # optional bookkeeping
        data.edge_index_phys = edge_index_phys
        data.A_phys = A_phys                 # (M,1) unscaled
        data.loads = loads                   # (N,d)

        # If you have your own assemble_bc(), keep it. Otherwise comment out / set zeros.
        bc_mask = torch.tensor(bc_mask_np, dtype=torch.float32)  # (N,d)
        try:
            data.fixed = bc_mask.bool()  # user-defined elsewhere
        except NameError:
            data.fixed = torch.zeros(N, d)       # fallback: no fixed DOFs info

        data.f_char = F_char.view(1)       # for inverse scaling later
        data.A_char = A_char
        data.I_char = I_char
        data.scale_y = S_y

        data_list.append(data)

    # ------------------ splits & loaders ------------------

    train_ds = data_list[:n_train]
    val_ds = data_list[n_train:n_train + n_val]
    test_ds = data_list[n_train + n_val:]

    bs = int(config['train']['batchsize'])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs)
    test_loader  = DataLoader(test_ds,  batch_size=1)

    return train_loader, val_loader, test_loader


def generate_planetruss_data_loader_var_load(args, config):

    # load the data
    mat = sio.loadmat(config['data']['datapath'])
    num_train_input = config['train']['num_train_input']
    u = np.stack(mat['u'][0])  # list of M elements
    v = np.stack(mat['v'][0])  # list of M elements
    coors = np.stack(mat['coors'][0])  # list of M elements
    par = np.stack(mat['input_var'][0])  # list of M elements
    ic_flag = np.stack(mat['total_ic_flag'][0])  # list of M elements

    # ------- globals (v7) -------
    globals_raw = {
        "A": load_global_field(mat, "A", stack=False),
        "E": load_global_field(mat, "E", stack=False),
        "Lx_ref": load_global_field(mat, "Lx", stack=False),
        "elems": load_global_field(mat, "conn", stack=False),
    }

    # ------------- check + convert globals to NumPy -------------
    required_names = ["A", "E", "Lx_ref", "elems"]

    for name in required_names:
        if globals_raw.get(name) is None:
            raise ValueError(f"Global '{name}' could not be loaded from MAT file.")
        globals_raw[name] = np.asarray(globals_raw[name])

    # unpack if you still like explicit names
    A = globals_raw["A"]
    E = globals_raw["E"]
    Lx_ref = globals_raw["Lx_ref"]
    elems_global = globals_raw["elems"]
    elems_np = np.asarray(elems_global, dtype=np.int64)  # (M, 2), 1-based
    elems = elems_np - 1

    F = np.stack(mat['loads'][0])

    '''
    prepare the data to support batchwise training
    '''
    # find the maximum number of nodes
    datasize = len(u)
    max_pde_nodes = 0
    max_par_nodes = 0
    max_bc_nodes = 0
    for i in range(datasize):
        num_pde = np.sum(1-ic_flag[i])
        if num_pde > max_pde_nodes:
            max_pde_nodes = num_pde
        num_par_ = par[i].shape[0]
        if num_par_ > max_par_nodes:
            max_par_nodes = num_par_
        num_bc = np.sum(ic_flag[i])
        if num_bc > max_bc_nodes:
            max_bc_nodes = num_bc
    max_pde_nodes = int(max_pde_nodes)
    max_bc_nodes = int(max_bc_nodes)
    max_par_nodes = int(max_par_nodes)

    # --- keep track of per-sample info only for TEST range
    test_orig_coors = []
    test_orig_u = []
    test_orig_v = []
    test_orig_par = []
    test_orig_A = []
    test_orig_loads = []

    test_norm_coors = []
    test_norm_u = []
    test_norm_v = []
    test_norm_parp = []
    test_flag = []

    test_meta = {
        "pde_idx": [], "bc_idx": [],
        "F_char": [], "A_char": [], "L_char": [], "S_y": []
        # store whatever normalize_truss_coords returns as 2nd item
    }

    # split the data
    # bar1 = [0,int(0.7*datasize)]
    # bar2 = [int(0.7*datasize),int(0.8*datasize)]
    # bar3 = [int(0.8*datasize),int(datasize)]
    bar1 = [0, num_train_input]
    bar2 = [num_train_input, num_train_input + 2000]
    bar3 = [num_train_input + 2000, num_train_input + 4000]

    # Precompute the integer range for quick membership test
    test_indices = set(range(bar3[0], min(bar3[1], datasize)))

    # append zeros to the data
    uT = []
    vT = []
    coorT = []
    parT = []
    par_flagT = []
    flagT = []
    S_yT = []
    for i in range(datasize):
        # extract the index of pde nodes and bc nodes
        pde_idx = np.where(ic_flag[i]==0)[1]
        bc_idx = np.where(ic_flag[i]==1)[1]
        num_pde = np.size(pde_idx)
        num_bc = np.size(bc_idx)
        # re-organize coors
        coorp = coors[i]
        coorp_norm, _ = normalize_truss_coords(coorp, Lx_ref)

        coorp_s = np.concatenate((coorp_norm[pde_idx, :], np.zeros((max_pde_nodes - num_pde, 2)), coorp_norm[bc_idx, :],
                                np.zeros((max_bc_nodes - num_bc, 2))), 0)  # (max_pde+max_bc,2)
        coorp_s = np.expand_dims(coorp_s, 0)  # (1,max_pde+max_bc,2)
        coorT.append(coorp_s)

        # re-organize solution
        up = u[i]
        vp = v[i]
        Ap = A
        loads = F[i]
        Fy = loads[:, 1]  # extract y-direction loads
        F_char = np.max(np.abs(Fy))  # take maximum magnitude of vertical load
        up_s, vp_s, S_y, A_char, L_char = scale_planetruss_displacements(coorp, Ap, F_char, E, up, vp, elem=elems)

        S_y = _canonicalize_Sy(S_y, loads.shape[1])  # (1,3)
        S_yT.append(S_y)

        up_s = np.concatenate((up_s[:,pde_idx], np.zeros((1,max_pde_nodes-num_pde)), up_s[:,bc_idx],
                             np.zeros((1,max_bc_nodes-num_bc))), -1)    # (1, max_pde+max_bc)
        uT.append(up_s)

        vp_s = np.concatenate((vp_s[:,pde_idx], np.zeros((1,max_pde_nodes - num_pde)), vp_s[:,bc_idx],
                             np.zeros((1,max_bc_nodes - num_bc))), -1)  # (1, max_pde+max_bc)
        vT.append(vp_s)


        # re-organize parameters
        parpv = par[i]
        load_nodes = tuple(np.where(Fy!=0)[0])
        parpv_norm = make_or_update_parp(coorp_norm, loads, F_char, parp=parpv, nodes=load_nodes, one_based=False)
        num_par = parpv_norm.shape[0]
        parp_s = np.concatenate((parpv_norm, np.zeros((max_par_nodes-num_par,3))), 0)    # (max_par,3)
        par_flag = np.concatenate((np.ones_like(parpv), np.zeros((max_par_nodes-num_par,3))), 0)    # (max_par,3)
        parp_s = np.expand_dims(parp_s, 0)    # (1,max_par,3)
        par_flag = np.expand_dims(par_flag, 0)    # (1,max_par,3)
        parT.append(parp_s)
        par_flagT.append(par_flag)
        # re-organize ic flag
        flagp = ic_flag[i]
        flagp = np.concatenate((flagp[:,pde_idx], -np.ones((1,max_pde_nodes-num_pde)),
                                flagp[:,bc_idx], -np.ones((1,max_bc_nodes-num_bc))), -1)    # (1, max_pde+max_bc)
        flagT.append(flagp)

        if i in test_indices:
            # originals (no padding, original ordering)
            test_orig_coors.append(coorp)
            test_orig_u.append(up)
            test_orig_v.append(vp)
            test_orig_par.append(parpv)
            test_orig_A.append(Ap)
            test_orig_loads.append(loads)

            # normalized + padded (the exact tensors fed to the model at test time)
            test_norm_coors.append(coorp_s[0])  # drop leading dim
            test_norm_u.append(up_s[0])
            test_norm_v.append(vp_s[0])
            test_norm_parp.append(parp_s[0])
            test_flag.append(flagp[0])

            # metadata for inverse-transform / reconstruction
            test_meta["pde_idx"].append(pde_idx.astype(np.int32))
            test_meta["bc_idx"].append(bc_idx.astype(np.int32))
            test_meta["F_char"].append(np.float64(F_char))
            test_meta["A_char"].append(np.asarray(A_char, dtype=np.float64))
            test_meta["L_char"].append(np.asarray(L_char, dtype=np.float64))
            test_meta["S_y"].append(np.asarray(S_y, dtype=np.float64))

    uT = np.concatenate(tuple(uT), 0)    # (M, max_node)
    vT = np.concatenate(tuple(vT), 0)  # (M, max_node)
    coorT = np.concatenate(tuple(coorT), 0)    # (M, max_node, 2)
    parT = np.concatenate(tuple(parT), 0)    # (M, max_par_nodes,3)
    flagT = np.concatenate(tuple(flagT), 0)    # (M, max_node)
    par_flagT = np.concatenate(tuple(par_flagT), 0)[:,:,0]    # (M, max_par_nodes)
    S_yT = np.concatenate(tuple(S_yT), 0)  # (M, 3)
    uT = torch.from_numpy(uT)
    vT = torch.from_numpy(vT)
    coorT = torch.from_numpy(coorT)
    parT = torch.from_numpy(parT)
    flagT = torch.from_numpy(flagT)
    par_flagT = torch.from_numpy(par_flagT)
    S_yT = torch.from_numpy(S_yT)


    train_dataset = torch.utils.data.TensorDataset(parT[bar1[0]:bar1[1],:,:], coorT[bar1[0]:bar1[1],:],
                                                   uT[bar1[0]:bar1[1],:], vT[bar1[0]:bar1[1],:],
                                                   flagT[bar1[0]:bar1[1],:], par_flagT[bar1[0]:bar1[1],:],
                                                   S_yT[bar1[0]:bar1[1],:])
    val_dataset = torch.utils.data.TensorDataset(parT[bar2[0]:bar2[1],:,:], coorT[bar2[0]:bar2[1],:],
                                                 uT[bar2[0]:bar2[1],:], vT[bar2[0]:bar2[1],:],
                                                 flagT[bar2[0]:bar2[1],:], par_flagT[bar2[0]:bar2[1],:],
                                                 S_yT[bar2[0]:bar2[1],:])
    test_dataset = torch.utils.data.TensorDataset(parT[bar3[0]:bar3[1],:,:], coorT[bar3[0]:bar3[1],:],
                                                  uT[bar3[0]:bar3[1],:], vT[bar3[0]:bar3[1],:],
                                                  flagT[bar3[0]:bar3[1],:], par_flagT[bar3[0]:bar3[1],:],
                                                  S_yT[bar3[0]:bar3[1],:])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batchsize'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['train']['batchsize'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # NOTE: the lists below were populated in your loop only for test indices:
    #   test_orig_coors, test_orig_u, test_orig_v, test_orig_par,
    #   test_norm_coors_s, test_norm_u_s, test_norm_v_s, test_norm_parp_s, test_norm_flag_s,
    #   test_meta[...]  (pde_idx, bc_idx, F_char, A_char, L_char, S_y, coord_norm_aux)

    mat_payload = {
        # Originals (ragged -> MATLAB cell arrays)
        "orig_coors": _as_cell(test_orig_coors),  # each cell: (Ni, 2)
        "orig_u": _as_cell(test_orig_u),  # each cell: (1, Ni)
        "orig_v": _as_cell(test_orig_v),  # each cell: (1, Ni)
        "orig_par": _as_cell(test_orig_par),  # each cell: (Pi, 3)
        "orig_A": _as_cell(test_orig_A),
        "orig_loads": _as_cell(test_orig_loads),

        # Normalized & padded (uniform -> numeric arrays)
        # Shapes in MATLAB: size(...)= [Nt, max_node, 2], etc.
        "norm_coors": np.stack(test_norm_coors, axis=0).astype(np.float64),
        "norm_u": np.stack(test_norm_u, axis=0).astype(np.float64),
        "norm_v": np.stack(test_norm_v, axis=0).astype(np.float64),
        "norm_parp": np.stack(test_norm_parp, axis=0).astype(np.float64),
        "flag": np.stack(test_flag, axis=0).astype(np.float64),

        # Metadata (ragged -> cell arrays; scalars -> numeric)
        "pde_idx": _as_cell([pi.astype(np.int32) for pi in test_meta["pde_idx"]]),
        "bc_idx": _as_cell([bi.astype(np.int32) for bi in test_meta["bc_idx"]]),
        "F_char": np.asarray(test_meta["F_char"], dtype=np.float64),  # (Nt,)

        # These may be scalars or arrays per sample -> store as cells
        "A_char": _as_cell(test_meta["A_char"]),
        "L_char": _as_cell(test_meta["L_char"]),
        "S_y": _as_cell(test_meta["S_y"]),

        # Helpful globals
        "max_pde_nodes": np.int32(max_pde_nodes),
        "max_bc_nodes": np.int32(max_bc_nodes),
        "max_par_nodes": np.int32(max_par_nodes),
        "test_start": np.int32(bar3[0]),
        "test_end": np.int32(min(bar3[1], datasize)),
        "E": np.asarray(E, dtype=np.float64),
        "elem": elems.astype(np.int32),
    }

    savemat(r'./res/saved_models/test_cache_{}_{}.mat'.format(args.data, args.model), mat_payload, do_compression=True)

    # store the number of nodes of different types
    num_nodes_list = (max_pde_nodes, max_bc_nodes, max_par_nodes)

    return train_loader, val_loader, test_loader, num_nodes_list


def generate_graphplanetruss_data_loader(args, config):
    # ------------------ splits ------------------
    n_train = int(config['train']['num_train'])
    n_val   = int(config['train']['num_val'])
    n_test  = int(config['train']['num_test'])
    total_num_data = n_train + n_val + n_test

    # ------------------ load .mat ------------------
    mat = sio.loadmat(config['data']['datapath'], squeeze_me=True, struct_as_record=False)
    graphs_mat = mat['graphs']

    # array of structs -> list
    if isinstance(graphs_mat, np.ndarray):
        graphs = list(graphs_mat.ravel())
    else:
        graphs = [graphs_mat]

    # ------- globals (v7) -------
    globals_raw = {
        "A": load_global_field(mat, "A", stack=False),
        "E": load_global_field(mat, "E", stack=False),
        "Lx_ref": load_global_field(mat, "Lx", stack=False),
        "elems": load_global_field(mat, "elements", stack=False),
        "bc": load_global_field(mat, "bc", stack=False),
    }

    # ------------- check + convert globals to NumPy -------------
    required_names = ["A", "E", "Lx_ref", "elems", "bc"]

    for name in required_names:
        if globals_raw.get(name) is None:
            raise ValueError(f"Global '{name}' could not be loaded from MAT file.")
        globals_raw[name] = np.asarray(globals_raw[name])

    # unpack if you still like explicit names
    A_global = globals_raw["A"]
    E_global = globals_raw["E"]
    Lx_ref_global = globals_raw["Lx_ref"]
    elems_global = globals_raw["elems"]
    bc_global = globals_raw["bc"]
    bc_global = bc_global.astype(np.int64)

    # ------------------ clip to requested total_num_data ------------------
    n_available = len(graphs)
    if total_num_data > n_available:
        print(f"[generate_graphplaneframe_data_loader] "
              f"Warning: requested {total_num_data} samples, "
              f"but MAT file only has {n_available}; truncating.")
        total_num_data = n_available

    graphs = graphs[:total_num_data]

    data_list = []
    for g in graphs:
        # -------- read fields (NumPy first) --------
        pos_np = np.asarray(g.nodes)  # (N, d)
        elems_np = np.asarray(elems_global, dtype=np.int64)  # (M, 2), 1-based
        loads_np = np.asarray(getattr(g, 'loads', np.zeros_like(pos_np)))
        bc_mask_np = np.asarray(bc_global)

        target = getattr(g, 'target', None)
        if target is None:
            raise ValueError("Missing 'target' in a case; Option A expects 'target' per graph.")
        y = torch.tensor(np.asarray(target), dtype=torch.float32)  # (N,d) or (M,1) or scalar

        # -------- torch tensors --------
        pos = torch.tensor(pos_np, dtype=torch.float32)            # (N,d)
        A_phys = torch.tensor(A_global, dtype=torch.float32)
        N, d = pos.shape

        # Edge index (0-based), then make undirected
        edge_index_phys = torch.from_numpy(elems_np - 1).t().contiguous()  # (2, M)

        # -------- E from global E (scalar or array -> scalar for scaling) --------
        A_char = scalar_from_global(A_global, name="A")
        E_char = scalar_from_global(E_global, name="E")
        Lx_ref_val = scalar_from_global(Lx_ref_global, name="Lx_ref")

        # to_undirected and duplicate A accordingly
        edge_index_gnn = to_undirected(
            edge_index_phys, num_nodes=N
        )

        # -------- geometry-derived edge features --------
        L, rel = edge_lengths(pos, edge_index_gnn)
        cs = rel / L.clamp_min(1e-9)  # [E,2] = (cosx, cosy)

        # typical (physical) length from *physical* edges
        L_phys, _ = edge_lengths(pos, edge_index_phys)
        L_char = L_phys.mean().clamp_min(1e-12)  # scalar, guard tiny

        # -------- normalize node coords --------
        # simple left-bottom anchoring and global scale
        x_left = pos[:, 0].min()
        y_bottom = pos[:, 1].min()
        pos_norm = torch.stack([(pos[:, 0] - x_left) / Lx_ref_val,
                                (pos[:, 1] - y_bottom) / Lx_ref_val], dim=1)

        # -------- loads: per-graph characteristic force for scaling --------
        loads = torch.tensor(loads_np, dtype=torch.float32)   # (N,d)
        Fy = loads[:, 1] if loads.shape[1] >= 2 else torch.zeros(N, dtype=torch.float32)
        F_char = Fy.abs().max()
        if not torch.isfinite(F_char) or F_char.item() == 0.0:
            F_char = torch.tensor(1.0, dtype=torch.float32)  # avoid divide-by-zero

        # node features (example): normalized coords + scaled vertical load
        Fy_scaled = (Fy / F_char).unsqueeze(1)  # (N,1)
        # x_node = pos_norm  # (N,2)
        x_node = torch.cat([pos_norm, Fy_scaled], dim=1).float()  # (N,3)

        # choose your edge_attr; here: just direction cosines
        edge_attr_gnn = torch.cat([cs, L/Lx_ref_val], dim=1)
        # edge_attr_gnn = cs

        # -------- target scaling (node-level assumed) --------
        S_y = (E_char * A_char) / (F_char * L_char)
        y_scaled = y * S_y

        # -------- build Data --------
        data = Data(
            x=x_node,                 # or x_node if you want loads too: x=x_node
            pos=pos_norm,               # positions used by message passing (normalized)
            pos_raw=pos,                # keep raw coords if your layers need them
            edge_index=edge_index_gnn,  # undirected
            edge_attr=edge_attr_gnn,    # [A/A_bar, L]
            y=y_scaled,                 # scaled targets
        )

        # optional bookkeeping
        data.edge_index_phys = edge_index_phys
        data.A_phys = A_phys                 # (M,1) unscaled
        data.loads = loads                   # (N,d)

        # If you have your own assemble_bc(), keep it. Otherwise comment out / set zeros.
        bc_mask = torch.tensor(bc_mask_np, dtype=torch.float32)  # (N,d)
        try:
            data.fixed = bc_mask.bool()  # user-defined elsewhere
        except NameError:
            data.fixed = torch.zeros(N, d)       # fallback: no fixed DOFs info

        data.f_char = F_char.view(1)       # for inverse scaling later
        data.A_char = A_char
        data.scale_y = S_y

        data_list.append(data)

    # ------------------ splits & loaders ------------------

    train_ds = data_list[:n_train]
    val_ds = data_list[n_train:n_train + n_val]
    test_ds = data_list[n_train + n_val:]

    bs = int(config['train']['batchsize'])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs)
    test_loader  = DataLoader(test_ds,  batch_size=1)

    return train_loader, val_loader, test_loader


def generate_spacetruss_data_loader_var_load(args, config):

    # load the data
    mat = sio.loadmat(config['data']['datapath'])
    num_train_input = config['train']['num_train_input']
    u = np.stack(mat['u'][0])   # list of M elements
    v = np.stack(mat['v'][0])    # list of M elements
    w = np.stack(mat['w'][0])  # list of M elements
    coors = np.stack(mat['coors'][0])    # list of M elements
    par = np.stack(mat['input_var'][0])    # list of M elements
    ic_flag = np.stack(mat['total_ic_flag'][0])   # list of M elements

    # ------- globals (v7) -------
    globals_raw = {
        "A": load_global_field(mat, "A", stack=False),
        "E": load_global_field(mat, "E", stack=False),
        "Lx_ref": load_global_field(mat, "Lx", stack=False),
        "elems": load_global_field(mat, "conn", stack=False),
    }

    # ------------- check + convert globals to NumPy -------------
    required_names = ["A", "E", "Lx_ref", "elems"]

    for name in required_names:
        if globals_raw.get(name) is None:
            raise ValueError(f"Global '{name}' could not be loaded from MAT file.")
        globals_raw[name] = np.asarray(globals_raw[name])

    # unpack if you still like explicit names
    A = globals_raw["A"]
    E = globals_raw["E"]
    Lx_ref = globals_raw["Lx_ref"]
    elems_global = globals_raw["elems"]
    elems_np = np.asarray(elems_global, dtype=np.int64)  # (M, 2), 1-based
    elems = elems_np - 1

    F = np.stack(mat['loads'][0])

    '''
    prepare the data to support batchwise training
    '''
    # find the maximum number of nodes
    datasize = len(u)
    max_pde_nodes = 0
    max_par_nodes = 0
    max_bc_nodes = 0
    for i in range(datasize):
        num_pde = np.sum(1-ic_flag[i])
        if num_pde > max_pde_nodes:
            max_pde_nodes = num_pde
        num_par_ = par[i].shape[0]
        if num_par_ > max_par_nodes:
            max_par_nodes = num_par_
        num_bc = np.sum(ic_flag[i])
        if num_bc > max_bc_nodes:
            max_bc_nodes = num_bc
    max_pde_nodes = int(max_pde_nodes)
    max_bc_nodes = int(max_bc_nodes)
    max_par_nodes = int(max_par_nodes)

    # --- keep track of per-sample info only for TEST range
    test_orig_coors = []
    test_orig_u = []
    test_orig_v = []
    test_orig_w = []
    test_orig_par = []
    test_orig_A = []
    test_orig_loads = []

    test_norm_coors = []
    test_norm_u = []
    test_norm_v = []
    test_norm_w = []
    test_norm_parp = []
    test_flag = []

    test_meta = {
        "pde_idx": [], "bc_idx": [],
        "F_char": [], "A_char": [], "L_char": [], "S_y": []
        # store whatever normalize_truss_coords returns as 2nd item
    }

    # split the data
    # bar1 = [0,int(0.7*datasize)]
    # bar2 = [int(0.7*datasize),int(0.8*datasize)]
    # bar3 = [int(0.8*datasize),int(datasize)]
    bar1 = [0, num_train_input]
    bar2 = [num_train_input, num_train_input + 2000]
    bar3 = [num_train_input + 2000, num_train_input + 4000]

    # Precompute the integer range for quick membership test
    test_indices = set(range(bar3[0], min(bar3[1], datasize)))

    # append zeros to the data
    uT = []
    vT = []
    wT = []
    coorT = []
    parT = []
    par_flagT = []
    flagT = []
    S_yT = []
    for i in range(datasize):
        # extract the index of pde nodes and bc nodes
        pde_idx = np.where(ic_flag[i]==0)[1]
        bc_idx = np.where(ic_flag[i]==1)[1]
        num_pde = np.size(pde_idx)
        num_bc = np.size(bc_idx)
        # re-organize coors
        coorp = coors[i]
        coorp_norm, _ = normalize_spacetruss_coords(coorp, Lx_ref)

        coorp_s = np.concatenate((coorp_norm[pde_idx, :], np.zeros((max_pde_nodes - num_pde, 3)), coorp_norm[bc_idx, :],
                                np.zeros((max_bc_nodes - num_bc, 3))), 0)  # (max_pde+max_bc,3)
        coorp_s = np.expand_dims(coorp_s, 0)  # (1,max_pde+max_bc,3)
        coorT.append(coorp_s)

        # re-organize solution
        up = u[i]
        vp = v[i]
        wp = w[i]
        Ap = A
        loads = F[i]
        Fz = loads[:, 2]  # extract z-direction loads
        F_char = np.max(np.abs(Fz))  # take maximum magnitude of vertical load
        up_s, vp_s, wp_s, S_y, A_char, L_char = scale_spacetruss_displacements(coorp, Ap, F_char, E, up, vp, wp, elem=elems)

        S_y = _canonicalize_Sy(S_y, loads.shape[1])  # (1,3)
        S_yT.append(S_y)

        up_s = np.concatenate((up_s[:,pde_idx], np.zeros((1,max_pde_nodes-num_pde)), up_s[:,bc_idx],
                             np.zeros((1,max_bc_nodes-num_bc))), -1)    # (1, max_pde+max_bc)
        uT.append(up_s)

        vp_s = np.concatenate((vp_s[:,pde_idx], np.zeros((1,max_pde_nodes - num_pde)), vp_s[:,bc_idx],
                             np.zeros((1,max_bc_nodes - num_bc))), -1)  # (1, max_pde+max_bc)
        vT.append(vp_s)

        wp_s = np.concatenate((wp_s[:, pde_idx], np.zeros((1, max_pde_nodes - num_pde)), wp_s[:, bc_idx],
                               np.zeros((1, max_bc_nodes - num_bc))), -1)  # (1, max_pde+max_bc)
        wT.append(wp_s)


        # re-organize parameters
        parpv = par[i]
        load_nodes = tuple(np.where(Fz!=0)[0])
        parpv_norm = make_or_update_parp3(coorp_norm, loads, F_char, parp=parpv, nodes=load_nodes, one_based=False)
        num_par = parpv_norm.shape[0]
        parp_s = np.concatenate((parpv_norm, np.zeros((max_par_nodes-num_par,4))), 0)    # (max_par,4)
        par_flag = np.concatenate((np.ones_like(parpv), np.zeros((max_par_nodes-num_par,4))), 0)    # (max_par,4)
        parp_s = np.expand_dims(parp_s, 0)    # (1,max_par,4)
        par_flag = np.expand_dims(par_flag, 0)    # (1,max_par,4)
        parT.append(parp_s)
        par_flagT.append(par_flag)
        # re-organize ic flag
        flagp = ic_flag[i]
        flagp = np.concatenate((flagp[:,pde_idx], -np.ones((1,max_pde_nodes-num_pde)),
                                flagp[:,bc_idx], -np.ones((1,max_bc_nodes-num_bc))), -1)    # (1, max_pde+max_bc)
        flagT.append(flagp)

        if i in test_indices:
            # originals (no padding, original ordering)
            test_orig_coors.append(coorp)
            test_orig_u.append(up)
            test_orig_v.append(vp)
            test_orig_w.append(wp)
            test_orig_par.append(parpv)
            test_orig_A.append(Ap)
            test_orig_loads.append(loads)

            # normalized + padded (the exact tensors fed to the model at test time)
            test_norm_coors.append(coorp_s[0])  # drop leading dim
            test_norm_u.append(up_s[0])
            test_norm_v.append(vp_s[0])
            test_norm_w.append(wp_s[0])
            test_norm_parp.append(parp_s[0])
            test_flag.append(flagp[0])

            # metadata for inverse-transform / reconstruction
            test_meta["pde_idx"].append(pde_idx.astype(np.int32))
            test_meta["bc_idx"].append(bc_idx.astype(np.int32))
            test_meta["F_char"].append(np.float64(F_char))
            test_meta["A_char"].append(np.asarray(A_char, dtype=np.float64))
            test_meta["L_char"].append(np.asarray(L_char, dtype=np.float64))
            test_meta["S_y"].append(np.asarray(S_y, dtype=np.float64))

    uT = np.concatenate(tuple(uT), 0)    # (M, max_node)
    vT = np.concatenate(tuple(vT), 0)  # (M, max_node)
    wT = np.concatenate(tuple(wT), 0)  # (M, max_node)
    coorT = np.concatenate(tuple(coorT), 0)    # (M, max_node, 3)
    parT = np.concatenate(tuple(parT), 0)    # (M, max_par_nodes,4)
    flagT = np.concatenate(tuple(flagT), 0)    # (M, max_node)
    par_flagT = np.concatenate(tuple(par_flagT), 0)[:,:,0]    # (M, max_par_nodes)
    S_yT = np.concatenate(tuple(S_yT), 0)  # (M, 3)
    uT = torch.from_numpy(uT)
    vT = torch.from_numpy(vT)
    wT = torch.from_numpy(wT)
    coorT = torch.from_numpy(coorT)
    parT = torch.from_numpy(parT)
    flagT = torch.from_numpy(flagT)
    par_flagT = torch.from_numpy(par_flagT)
    S_yT = torch.from_numpy(S_yT)


    train_dataset = torch.utils.data.TensorDataset(parT[bar1[0]:bar1[1],:,:], coorT[bar1[0]:bar1[1],:],
                                                   uT[bar1[0]:bar1[1],:], vT[bar1[0]:bar1[1],:], wT[bar1[0]:bar1[1],:],
                                                   flagT[bar1[0]:bar1[1],:], par_flagT[bar1[0]:bar1[1],:],
                                                   S_yT[bar1[0]:bar1[1],:])
    val_dataset = torch.utils.data.TensorDataset(parT[bar2[0]:bar2[1],:,:], coorT[bar2[0]:bar2[1],:],
                                                 uT[bar2[0]:bar2[1],:], vT[bar2[0]:bar2[1],:], wT[bar2[0]:bar2[1],:],
                                                 flagT[bar2[0]:bar2[1],:], par_flagT[bar2[0]:bar2[1],:],
                                                 S_yT[bar2[0]:bar2[1],:])
    test_dataset = torch.utils.data.TensorDataset(parT[bar3[0]:bar3[1],:,:], coorT[bar3[0]:bar3[1],:],
                                                  uT[bar3[0]:bar3[1],:], vT[bar3[0]:bar3[1],:], wT[bar3[0]:bar3[1],:],
                                                  flagT[bar3[0]:bar3[1],:], par_flagT[bar3[0]:bar3[1],:],
                                                  S_yT[bar3[0]:bar3[1],:])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batchsize'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['train']['batchsize'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # NOTE: the lists below were populated in your loop only for test indices:
    #   test_orig_coors, test_orig_u, test_orig_v, test_orig_par,
    #   test_norm_coors_s, test_norm_u_s, test_norm_v_s, test_norm_parp_s, test_norm_flag_s,
    #   test_meta[...]  (pde_idx, bc_idx, F_char, A_char, L_char, S_y, coord_norm_aux)

    mat_payload = {
        # Originals (ragged -> MATLAB cell arrays)
        "orig_coors": _as_cell(test_orig_coors),  # each cell: (Ni, 2)
        "orig_u": _as_cell(test_orig_u),  # each cell: (1, Ni)
        "orig_v": _as_cell(test_orig_v),  # each cell: (1, Ni)
        "orig_w": _as_cell(test_orig_w),  # each cell: (1, Ni)
        "orig_par": _as_cell(test_orig_par),  # each cell: (Pi, 3)
        "orig_A": _as_cell(test_orig_A),
        "orig_loads": _as_cell(test_orig_loads),

        # Normalized & padded (uniform -> numeric arrays)
        # Shapes in MATLAB: size(...)= [Nt, max_node, 2], etc.
        "norm_coors": np.stack(test_norm_coors, axis=0).astype(np.float64),
        "norm_u": np.stack(test_norm_u, axis=0).astype(np.float64),
        "norm_v": np.stack(test_norm_v, axis=0).astype(np.float64),
        "norm_w": np.stack(test_norm_w, axis=0).astype(np.float64),
        "norm_parp": np.stack(test_norm_parp, axis=0).astype(np.float64),
        "flag": np.stack(test_flag, axis=0).astype(np.float64),

        # Metadata (ragged -> cell arrays; scalars -> numeric)
        "pde_idx": _as_cell([pi.astype(np.int32) for pi in test_meta["pde_idx"]]),
        "bc_idx": _as_cell([bi.astype(np.int32) for bi in test_meta["bc_idx"]]),
        "F_char": np.asarray(test_meta["F_char"], dtype=np.float64),  # (Nt,)

        # These may be scalars or arrays per sample -> store as cells
        "A_char": _as_cell(test_meta["A_char"]),
        "L_char": _as_cell(test_meta["L_char"]),
        "S_y": _as_cell(test_meta["S_y"]),

        # Helpful globals
        "max_pde_nodes": np.int32(max_pde_nodes),
        "max_bc_nodes": np.int32(max_bc_nodes),
        "max_par_nodes": np.int32(max_par_nodes),
        "test_start": np.int32(bar3[0]),
        "test_end": np.int32(min(bar3[1], datasize)),
        "E": np.asarray(E, dtype=np.float64),
        "elem": elems.astype(np.int32),
    }

    savemat(r'./res/saved_models/test_cache_{}_{}.mat'.format(args.data, args.model), mat_payload, do_compression=True)

    # store the number of nodes of different types
    num_nodes_list = (max_pde_nodes, max_bc_nodes, max_par_nodes)

    return train_loader, val_loader, test_loader, num_nodes_list


def generate_graphgridspacetruss_data_loader(args, config):
    # ------------------ splits ------------------
    n_train = int(config['train']['num_train'])
    n_val   = int(config['train']['num_val'])
    n_test  = int(config['train']['num_test'])
    total_num_data = n_train + n_val + n_test

    # ------------------ load .mat (v7 or v7.3) ------------------
    datapath = config['data']['datapath']

    # ------------------ load .mat (v7 or v7.3) ------------------
    try:
        # Try standard v7 MAT-file first
        mat = sio.loadmat(datapath, squeeze_me=True, struct_as_record=False)
        graphs_mat = mat['graphs']

        globals_raw = {
            "A": load_global_field(mat, "A", stack=False),
            "E": load_global_field(mat, "E", stack=False),
            "Lx_ref": load_global_field(mat, "Lx", stack=False),
            "elems": load_global_field(mat, "elements", stack=False),
            "bc": load_global_field(mat, "bc", stack=False),
        }

        if isinstance(graphs_mat, np.ndarray):
            graphs = list(graphs_mat.ravel())  # array of structs -> list
        else:
            graphs = [graphs_mat]

    except NotImplementedError:
        # v7.3 fallback: mat73 gives dict-of-arrays for a 1x30000 struct
        mat = mat73.loadmat(datapath)
        graphs_raw = mat['graphs']

        globals_raw = {
            "A": load_global_field(mat, "A", stack=False),
            "E": load_global_field(mat, "E", stack=False),
            "Lx_ref": load_global_field(mat, "Lx", stack=False),
            "elems": load_global_field(mat, "elements", stack=False),
            "bc": load_global_field(mat, "bc", stack=False),
        }

        # Case 1: dict-of-arrays (what you printed)
        if isinstance(graphs_raw, dict):
            # pick any field to detect number of graphs
            some_field = next(iter(graphs_raw.values()))
            if isinstance(some_field, list):
                n_graphs = len(some_field)
            elif isinstance(some_field, np.ndarray):
                n_graphs = some_field.shape[0]
            else:
                raise TypeError(
                    f"Unsupported type for field array in v7.3 graphs: {type(some_field)}"
                )

            graphs = []
            for i in range(n_graphs):
                d = {}
                for k, v in graphs_raw.items():
                    # v is list/array over graphs
                    d[k] = v[i]
                graphs.append(SimpleNamespace(**d))

        # Case 2: already a list of dicts/objects
        elif isinstance(graphs_raw, list):
            graphs = [
                SimpleNamespace(**g) if isinstance(g, dict) else g
                for g in graphs_raw
            ]

        # Fallback: single struct-like object
        else:
            graphs = [graphs_raw]

    # ------------- check + convert globals to NumPy -------------
    required_names = ["A", "E", "Lx_ref", "elems", "bc"]

    for name in required_names:
        if globals_raw.get(name) is None:
            raise ValueError(f"Global '{name}' could not be loaded from MAT file.")
        globals_raw[name] = np.asarray(globals_raw[name])

    # unpack if you still like explicit names
    A_global = globals_raw["A"]
    E_global = globals_raw["E"]
    Lx_ref_global = globals_raw["Lx_ref"]
    elems_global = globals_raw["elems"]
    bc_global = globals_raw["bc"]
    bc_global = bc_global.astype(np.int64)

    # ------------------ clip to requested total_num_data ------------------
    n_available = len(graphs)
    if total_num_data > n_available:
        print(f"[generate_graphgridspacetruss_data_loader] "
              f"Warning: requested {total_num_data} samples, "
              f"but MAT file only has {n_available}; truncating.")
        total_num_data = n_available

    graphs = graphs[:total_num_data]

    data_list = []
    for g in graphs:
        # -------- read fields (NumPy first) --------
        pos_np = np.asarray(g.nodes)  # (N, d)
        elems_np = np.asarray(elems_global, dtype=np.int64)  # (M, 2), 1-based
        loads_np = np.asarray(getattr(g, 'loads', np.zeros_like(pos_np)))
        bc_mask_np = np.asarray(bc_global)

        target = getattr(g, 'target', None)
        if target is None:
            raise ValueError("Missing 'target' in a case; Option A expects 'target' per graph.")
        y = torch.tensor(np.asarray(target), dtype=torch.float32)  # (N,d) or (M,1) or scalar

        # -------- torch tensors --------
        pos = torch.tensor(pos_np, dtype=torch.float32)  # (N,d)
        A_phys = torch.tensor(A_global, dtype=torch.float32)
        N, d = pos.shape

        # Edge index (0-based), then make undirected
        edge_index_phys = torch.from_numpy(elems_np - 1).t().contiguous()  # (2, M)

        # -------- E from global E (scalar or array -> scalar for scaling) --------
        A_char = scalar_from_global(A_global, name="A")
        E_char = scalar_from_global(E_global, name="E")
        Lx_ref_val = scalar_from_global(Lx_ref_global, name="Lx_ref")

        # to_undirected and duplicate A accordingly
        edge_index_gnn = to_undirected(
            edge_index_phys, num_nodes=N
        )

        # -------- geometry-derived edge features --------
        L, rel = edge_lengths(pos, edge_index_gnn)
        cs = rel / L.clamp_min(1e-9)  # [E,2] = (cosx, cosy)

        # typical (physical) length from *physical* edges
        L_phys, _ = edge_lengths(pos, edge_index_phys)
        L_char = L_phys.mean().clamp_min(1e-12)  # scalar, guard tiny

        # -------- normalize node coords --------
        # simple left-bottom anchoring and global scale
        x_left = pos[:, 0].min()
        y_bottom = pos[:, 1].min()
        z_bottom = pos[:, 2].min()
        pos_norm = torch.stack([(pos[:, 0] - x_left) / Lx_ref_val,
                                (pos[:, 1] - y_bottom) / Lx_ref_val,
                                (pos[:, 2] - z_bottom) / Lx_ref_val], dim=1)

        # -------- loads: per-graph characteristic force for scaling --------
        loads = torch.tensor(loads_np, dtype=torch.float32)   # (N,d)
        if loads.shape[1] < 3:
            # pad loads to 3D for consistent norm (ok if truly 2D)
            loads = torch.cat([loads, torch.zeros(N, 3 - loads.shape[1])], dim=1)
        Fz = loads[:, 2]
        F_char = Fz.abs().max()
        if (not torch.isfinite(F_char)) or F_char.item() == 0.0:
            F_char = torch.tensor(1.0, dtype=torch.float32)  # robust fallback

        # node features (example): normalized coords + scaled vertical load
        Fz_scaled = (Fz / F_char).unsqueeze(1)  # (N,1)
        # x_node = pos_norm   # (N, 2)
        x_node = torch.cat([pos_norm, Fz_scaled], dim=1).float()  # (N,4)

        # edge_attr_gnn = torch.cat([A_scaled, cs], dim=1)  # e.g., [A/A_bar, |r|]
        edge_attr_gnn = torch.cat([cs, L/Lx_ref_val], dim=1)

        # -------- target scaling (node-level assumed) --------
        S_y = (E_char * A_char) / (F_char * L_char)
        y_scaled = y * S_y

        # -------- build Data --------
        data = Data(
            x=x_node,                 # or x_node if you want loads too: x=x_node
            pos=pos_norm,               # positions used by message passing (normalized)
            pos_raw=pos,                # keep raw coords if your layers need them
            edge_index=edge_index_gnn,  # undirected
            edge_attr=edge_attr_gnn,    # [A/A_bar, L]
            y=y_scaled,                 # scaled targets
        )

        # optional bookkeeping
        data.edge_index_phys = edge_index_phys
        data.A_phys = A_phys                 # (M,1) unscaled
        data.loads = loads                   # (N,d)
        # If you have your own assemble_bc(), keep it. Otherwise comment out / set zeros.
        bc_mask = torch.tensor(bc_mask_np, dtype=torch.float32)  # (N,d)
        try:
            data.fixed = bc_mask.bool()  # user-defined elsewhere            d
        except NameError:
            data.fixed = torch.zeros(N, d)       # fallback: no fixed DOFs info

        data.f_char = F_char.view(1)       # for inverse scaling later
        data.A_char = A_char
        data.scale_y = S_y

        data_list.append(data)

    # ------------------ splits & loaders ------------------

    train_ds = data_list[:n_train]
    val_ds = data_list[n_train:n_train + n_val]
    test_ds = data_list[n_train + n_val:]

    bs = int(config['train']['batchsize'])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs)
    test_loader  = DataLoader(test_ds,  batch_size=1)

    return train_loader, val_loader, test_loader


def generate_spaceframe_data_loader(args, config):

    # load the data
    mat = sio.loadmat(config['data']['datapath'])
    num_train_input = config['train']['num_train_input']
    u = mat['u'][0]   # list of M elements
    v = mat['v'][0]   # list of M elements
    w = mat['w'][0]   # list of M elements
    thx = mat['thx'][0]  # list of M elements
    thy = mat['thy'][0]  # list of M elements
    thz = mat['thz'][0]  # list of M elements
    coors = mat['coors'][0]    # list of M elements
    par = mat['input_par'][0]    # list of M elements
    ic_flag = mat['varycoor_flag'][0]   # list of M elements

    # ------- globals (v7) -------
    globals_raw = {
        "A": load_global_field(mat, "A", stack=True),
        "Iy": load_global_field(mat, "Iy", stack=True),
        "Iz": load_global_field(mat, "Iz", stack=True),
        "J": load_global_field(mat, "J", stack=True),
        "E": load_global_field(mat, "E", stack=False),
        "G": load_global_field(mat, "G", stack=False),
        "Lx_ref": load_global_field(mat, "Lx", stack=False),
        "elems": load_global_field(mat, "conn", stack=False),
    }

    # ------------- check + convert globals to NumPy -------------
    required_names = ["A", "E", "G", "Iy", "Iz", "J", "Lx_ref", "elems"]

    for name in required_names:
        if globals_raw.get(name) is None:
            raise ValueError(f"Global '{name}' could not be loaded from MAT file.")
        globals_raw[name] = np.asarray(globals_raw[name])

    # unpack if you still like explicit names
    A = globals_raw["A"]
    E = globals_raw["E"]
    G = globals_raw["G"]
    Iy = globals_raw["Iy"]
    Iz = globals_raw["Iz"]
    J = globals_raw["J"]
    Lx_ref = globals_raw["Lx_ref"]
    elems_global = globals_raw["elems"]  # NEW
    elems_np = np.asarray(elems_global, dtype=np.int64)  # (M, 2), 1-based
    elems = elems_np - 1

    F = np.stack(mat['loads'][0])

    '''
    prepare the data to support batchwise training
    '''
    # find the maximum number of nodes
    datasize = len(u)
    # datasize = num_train_input
    max_pde_nodes = 0
    max_par_nodes = 0
    max_bc_nodes = 0
    for i in range(datasize):
        num_pde = np.sum(1-ic_flag[i])
        if num_pde > max_pde_nodes:
            max_pde_nodes = num_pde
        num_par_ = par[i].shape[0]
        if num_par_ > max_par_nodes:
            max_par_nodes = num_par_
        num_bc = np.sum(ic_flag[i])
        if num_bc > max_bc_nodes:
            max_bc_nodes = num_bc
    max_pde_nodes = int(max_pde_nodes)
    max_bc_nodes = int(max_bc_nodes)
    max_par_nodes = int(max_par_nodes)

    # --- keep track of per-sample info only for TEST range
    test_orig_coors = []
    test_orig_u = []
    test_orig_v = []
    test_orig_w = []
    test_orig_thx = []
    test_orig_thy = []
    test_orig_thz = []
    test_orig_par = []
    test_orig_A = []
    test_orig_loads = []

    test_norm_coors = []
    test_norm_u = []
    test_norm_v = []
    test_norm_w = []
    test_norm_thx = []
    test_norm_thy = []
    test_norm_thz = []
    test_norm_parp = []
    test_flag = []

    test_meta = {
        "pde_idx": [], "bc_idx": [],
        "F_char": [], "A_char": [], "Iy_char": [], "Iz_char": [], "J_char": [], "L_char": [], "S_y": []
        # store whatever normalize_truss_coords returns as 2nd item
    }

    # split the data
    # bar1 = [0,int(0.7*datasize)]
    # bar2 = [int(0.7*datasize),int(0.8*datasize)]
    # bar3 = [int(0.8*datasize),int(datasize)]
    bar1 = [0, num_train_input]
    bar2 = [num_train_input, num_train_input + 2000]
    bar3 = [num_train_input + 2000, num_train_input + 4000]

    # Precompute the integer range for quick membership test
    test_indices = set(range(bar3[0], min(bar3[1], datasize)))

    # append zeros to the data
    uT = []
    vT = []
    wT = []
    thxT = []
    thyT = []
    thzT = []
    coorT = []
    parT = []
    par_flagT = []
    flagT = []
    S_yT = []
    for i in range(datasize):
        # extract the index of pde nodes and bc nodes
        pde_idx = np.where(ic_flag[i]==0)[1]
        bc_idx = np.where(ic_flag[i]==1)[1]
        num_pde = np.size(pde_idx)
        num_bc = np.size(bc_idx)
        # re-organize coors
        coorp = coors[i]
        coorp_norm, _ = normalize_spacetruss_coords(coorp, Lx_ref)

        coorp_s = np.concatenate((coorp_norm[pde_idx, :], np.zeros((max_pde_nodes - num_pde, 3)), coorp_norm[bc_idx, :],
                                  np.zeros((max_bc_nodes - num_bc, 3))), 0)  # (max_pde+max_bc,3)
        coorp_s = np.expand_dims(coorp_s, 0)  # (1,max_pde+max_bc,3)
        coorT.append(coorp_s)

        # re-organize solution
        up = u[i]
        vp = v[i]
        wp = w[i]
        thxp = thx[i]
        thyp = thy[i]
        thzp = thz[i]
        Ap = A
        loads = F[i]
        Fz = loads[:, 2]  # extract z-direction loads
        F_char = np.max(np.abs(Fz))  # take maximum magnitude of vertical load
        up_s, vp_s, wp_s, thxp_s, thyp_s, thzp_s, S_y, A_char, Iy_char, Iz_char, J_char, L_char \
            = scale_spaceframe_displacements(coorp, Ap, Iy, Iz, J, F_char, E, G, up, vp, wp, thxp, thyp, thzp,
                                                                               elem=elems)

        S_y = _canonicalize_Sy(S_y, loads.shape[1])  # (1, 6)
        S_yT.append(S_y)

        up_s = np.concatenate((up_s[:, pde_idx], np.zeros((1, max_pde_nodes - num_pde)), up_s[:, bc_idx],
                               np.zeros((1, max_bc_nodes - num_bc))), -1)  # (1, max_pde+max_bc)
        uT.append(up_s)

        vp_s = np.concatenate((vp_s[:, pde_idx], np.zeros((1, max_pde_nodes - num_pde)), vp_s[:, bc_idx],
                               np.zeros((1, max_bc_nodes - num_bc))), -1)  # (1, max_pde+max_bc)
        vT.append(vp_s)

        wp_s = np.concatenate((wp_s[:, pde_idx], np.zeros((1, max_pde_nodes - num_pde)), wp_s[:, bc_idx],
                               np.zeros((1, max_bc_nodes - num_bc))), -1)  # (1, max_pde+max_bc)
        wT.append(wp_s)

        thxp_s = np.concatenate((thxp_s[:, pde_idx], np.zeros((1, max_pde_nodes - num_pde)), thxp_s[:, bc_idx],
                             np.zeros((1, max_bc_nodes - num_bc))), -1)  # (1, max_pde+max_bc)
        thxT.append(thxp_s)

        thyp_s = np.concatenate((thyp_s[:, pde_idx], np.zeros((1, max_pde_nodes - num_pde)), thyp_s[:, bc_idx],
                                np.zeros((1, max_bc_nodes - num_bc))), -1)  # (1, max_pde+max_bc)
        thyT.append(thyp_s)

        thzp_s = np.concatenate((thzp_s[:, pde_idx], np.zeros((1, max_pde_nodes - num_pde)), thzp_s[:, bc_idx],
                                np.zeros((1, max_bc_nodes - num_bc))), -1)  # (1, max_pde+max_bc)
        thzT.append(thzp_s)

        # re-organize parameters
        parpv = par[i]
        load_nodes = tuple(np.where(Fz!=0)[0])
        parpv_norm = make_or_update_parp3(coorp_norm, loads, F_char, parp=parpv, nodes=load_nodes, one_based=False)
        num_par = parpv_norm.shape[0]
        parp_s = np.concatenate((parpv_norm, np.zeros((max_par_nodes - num_par, 4))), 0)  # (max_par,4)
        par_flag = np.concatenate((np.ones_like(parpv), np.zeros((max_par_nodes - num_par, 4))), 0)  # (max_par,4)
        parp_s = np.expand_dims(parp_s, 0)  # (1,max_par,4)
        par_flag = np.expand_dims(par_flag, 0)  # (1,max_par,4)
        parT.append(parp_s)
        par_flagT.append(par_flag)
        # re-organize ic flag
        flagp = ic_flag[i]
        flagp = np.concatenate((flagp[:,pde_idx], -np.ones((1,max_pde_nodes-num_pde)), flagp[:,bc_idx],
                                -np.ones((1,max_bc_nodes-num_bc))), -1)    # (1, max_pde+max_bc)
        flagT.append(flagp)

        if i in test_indices:
            # originals (no padding, original ordering)
            test_orig_coors.append(coorp)
            test_orig_u.append(up)
            test_orig_v.append(vp)
            test_orig_w.append(wp)
            test_orig_thx.append(thxp)
            test_orig_thy.append(thyp)
            test_orig_thz.append(thzp)
            test_orig_par.append(parpv)
            test_orig_A.append(Ap)
            test_orig_loads.append(loads)

            # normalized + padded (the exact tensors fed to the model at test time)
            test_norm_coors.append(coorp_s[0])  # drop leading dim
            test_norm_u.append(up_s[0])
            test_norm_v.append(vp_s[0])
            test_norm_w.append(wp_s[0])
            test_norm_thx.append(thxp_s[0])
            test_norm_thy.append(thyp_s[0])
            test_norm_thz.append(thzp_s[0])
            test_norm_parp.append(parp_s[0])
            test_flag.append(flagp[0])

            # metadata for inverse-transform / reconstruction
            test_meta["pde_idx"].append(pde_idx.astype(np.int32))
            test_meta["bc_idx"].append(bc_idx.astype(np.int32))
            test_meta["F_char"].append(np.float64(F_char))
            test_meta["A_char"].append(np.asarray(A_char, dtype=np.float64))
            test_meta["Iy_char"].append(np.asarray(Iy_char, dtype=np.float64))
            test_meta["Iz_char"].append(np.asarray(Iz_char, dtype=np.float64))
            test_meta["J_char"].append(np.asarray(J_char, dtype=np.float64))
            test_meta["L_char"].append(np.asarray(L_char, dtype=np.float64))
            test_meta["S_y"].append(np.asarray(S_y, dtype=np.float64))

    uT = np.concatenate(tuple(uT), 0)    # (M, max_node)
    vT = np.concatenate(tuple(vT), 0)  # (M, max_node)
    wT = np.concatenate(tuple(wT), 0)  # (M, max_node)
    thxT = np.concatenate(tuple(thxT), 0)  # (M, max_node)
    thyT = np.concatenate(tuple(thyT), 0)  # (M, max_node)
    thzT = np.concatenate(tuple(thzT), 0)  # (M, max_node)
    coorT = np.concatenate(tuple(coorT), 0)    # (M, max_node, 3)
    parT = np.concatenate(tuple(parT), 0)    # (M, max_par_nodes,4)
    flagT = np.concatenate(tuple(flagT), 0)    # (M, max_node)
    par_flagT = np.concatenate(tuple(par_flagT), 0)[:,:,0]    # (M, max_par_nodes)
    S_yT = np.concatenate(tuple(S_yT), 0)  # (M, 6)
    uT = torch.from_numpy(uT)
    vT = torch.from_numpy(vT)
    wT = torch.from_numpy(wT)
    thxT = torch.from_numpy(thxT)
    thyT = torch.from_numpy(thyT)
    thzT = torch.from_numpy(thzT)
    coorT = torch.from_numpy(coorT)
    parT = torch.from_numpy(parT)
    flagT = torch.from_numpy(flagT)
    par_flagT = torch.from_numpy(par_flagT)
    S_yT = torch.from_numpy(S_yT)


    train_dataset = torch.utils.data.TensorDataset(parT[bar1[0]:bar1[1],:,:],
            coorT[bar1[0]:bar1[1],:], uT[bar1[0]:bar1[1],:], vT[bar1[0]:bar1[1],:], wT[bar1[0]:bar1[1],:],
                            thxT[bar1[0]:bar1[1],:], thyT[bar1[0]:bar1[1],:], thzT[bar1[0]:bar1[1],:],
                           flagT[bar1[0]:bar1[1],:], par_flagT[bar1[0]:bar1[1],:], S_yT[bar1[0]:bar1[1],:])
    val_dataset = torch.utils.data.TensorDataset(parT[bar2[0]:bar2[1],:,:],
            coorT[bar2[0]:bar2[1],:], uT[bar2[0]:bar2[1],:], vT[bar2[0]:bar2[1],:], wT[bar2[0]:bar2[1],:],
                            thxT[bar2[0]:bar2[1],:], thyT[bar2[0]:bar2[1],:], thzT[bar2[0]:bar2[1],:],
                          flagT[bar2[0]:bar2[1],:], par_flagT[bar2[0]:bar2[1],:], S_yT[bar2[0]:bar2[1],:])
    test_dataset = torch.utils.data.TensorDataset(parT[bar3[0]:bar3[1],:,:],
            coorT[bar3[0]:bar3[1],:], uT[bar3[0]:bar3[1],:], vT[bar3[0]:bar3[1],:], wT[bar3[0]:bar3[1],:],
                            thxT[bar3[0]:bar3[1],:], thyT[bar3[0]:bar3[1],:], thzT[bar3[0]:bar3[1],:],
                          flagT[bar3[0]:bar3[1],:], par_flagT[bar3[0]:bar3[1],:], S_yT[bar3[0]:bar3[1],:])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batchsize'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['train']['batchsize'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # NOTE: the lists below were populated in your loop only for test indices:
    #   test_orig_coors, test_orig_u, test_orig_v, test_orig_par,
    #   test_norm_coors_s, test_norm_u_s, test_norm_v_s, test_norm_parp_s, test_norm_flag_s,
    #   test_meta[...]  (pde_idx, bc_idx, F_char, A_char, L_char, S_y, coord_norm_aux)

    mat_payload = {
        # Originals (ragged -> MATLAB cell arrays)
        "orig_coors": _as_cell(test_orig_coors),  # each cell: (Ni, 2)
        "orig_u": _as_cell(test_orig_u),  # each cell: (1, Ni)
        "orig_v": _as_cell(test_orig_v),  # each cell: (1, Ni)
        "orig_w": _as_cell(test_orig_w),  # each cell: (1, Ni)
        "orig_thx": _as_cell(test_orig_thx),  # each cell: (1, Ni)
        "orig_thy": _as_cell(test_orig_thy),  # each cell: (1, Ni)
        "orig_thz": _as_cell(test_orig_thz),  # each cell: (1, Ni)
        "orig_par": _as_cell(test_orig_par),  # each cell: (Pi, 3)
        "orig_A": _as_cell(test_orig_A),
        "orig_loads": _as_cell(test_orig_loads),

        # Normalized & padded (uniform -> numeric arrays)
        # Shapes in MATLAB: size(...)= [Nt, max_node, 2], etc.
        "norm_coors": np.stack(test_norm_coors, axis=0).astype(np.float64),
        "norm_u": np.stack(test_norm_u, axis=0).astype(np.float64),
        "norm_v": np.stack(test_norm_v, axis=0).astype(np.float64),
        "norm_w": np.stack(test_norm_w, axis=0).astype(np.float64),
        "norm_thx": np.stack(test_norm_thx, axis=0).astype(np.float64),
        "norm_thy": np.stack(test_norm_thy, axis=0).astype(np.float64),
        "norm_thz": np.stack(test_norm_thz, axis=0).astype(np.float64),
        "norm_parp": np.stack(test_norm_parp, axis=0).astype(np.float64),
        "flag": np.stack(test_flag, axis=0).astype(np.float64),

        # Metadata (ragged -> cell arrays; scalars -> numeric)
        "pde_idx": _as_cell([pi.astype(np.int32) for pi in test_meta["pde_idx"]]),
        "bc_idx": _as_cell([bi.astype(np.int32) for bi in test_meta["bc_idx"]]),
        "F_char": np.asarray(test_meta["F_char"], dtype=np.float64),  # (Nt,)

        # These may be scalars or arrays per sample -> store as cells
        "A_char": _as_cell(test_meta["A_char"]),
        "Iy_char": _as_cell(test_meta["Iy_char"]),
        "Iz_char": _as_cell(test_meta["Iz_char"]),
        "J_char": _as_cell(test_meta["J_char"]),
        "L_char": _as_cell(test_meta["L_char"]),
        "S_y": _as_cell(test_meta["S_y"]),

        # Helpful globals
        "max_pde_nodes": np.int32(max_pde_nodes),
        "max_bc_nodes": np.int32(max_bc_nodes),
        "max_par_nodes": np.int32(max_par_nodes),
        "test_start": np.int32(bar3[0]),
        "test_end": np.int32(min(bar3[1], datasize)),
        "E": np.asarray(E, dtype=np.float64),
        "G": np.asarray(G, dtype=np.float64),
        "elem": elems.astype(np.int32),
    }

    savemat(r'./res/saved_models/test_cache_{}_{}.mat'.format(args.data, args.model), mat_payload, do_compression=True)

    # store the number of nodes of different types
    num_nodes_list = (max_pde_nodes, max_bc_nodes, max_par_nodes)

    return train_loader, val_loader, test_loader, num_nodes_list


def generate_graphspaceframe_data_loader(args, config):
    # ------------------ splits ------------------
    n_train = int(config['train']['num_train'])
    n_val = int(config['train']['num_val'])
    n_test = int(config['train']['num_test'])
    total_num_data = n_train + n_val + n_test

    # ------------------ load .mat (v7 or v7.3) ------------------
    datapath = config['data']['datapath']

    try:
        # Try standard v7 MAT-file first
        mat = sio.loadmat(datapath, squeeze_me=True, struct_as_record=False)
        graphs_mat = mat['graphs']

        # ------- globals (v7) -------
        globals_raw = {
            "A":      load_global_field(mat, "A",  stack=True),
            "Iy":     load_global_field(mat, "Iy", stack=True),
            "Iz":     load_global_field(mat, "Iz", stack=True),
            "J":      load_global_field(mat, "J",  stack=True),
            "E":      load_global_field(mat, "E",  stack=False),
            "G":      load_global_field(mat, "G",  stack=False),
            "Lx_ref": load_global_field(mat, "Lx", stack=False),
            "elems":  load_global_field(mat, "elements", stack=False),
            "bc":     load_global_field(mat, "bc",       stack=False),
        }


        if isinstance(graphs_mat, np.ndarray):
            graphs = list(graphs_mat.ravel())  # array of structs -> list
        else:
            graphs = [graphs_mat]

    except NotImplementedError:
        # v7.3 fallback: mat73 gives dict-of-arrays for a 1xN struct
        mat = mat73.loadmat(datapath)
        graphs_raw = mat['graphs']

        # ------- globals (v7.3) -------
        globals_raw = {
            "A": load_global_field(mat, "A", stack=True),
            "Iy": load_global_field(mat, "Iy", stack=True),
            "Iz": load_global_field(mat, "Iz", stack=True),
            "J": load_global_field(mat, "J", stack=True),
            "E": load_global_field(mat, "E", stack=False),
            "G": load_global_field(mat, "G", stack=False),
            "Lx_ref": load_global_field(mat, "Lx", stack=False),
            "elems": load_global_field(mat, "elements", stack=False),
            "bc": load_global_field(mat, "bc", stack=False),
        }

        # Case 1: dict-of-arrays
        if isinstance(graphs_raw, dict):
            some_field = next(iter(graphs_raw.values()))
            if isinstance(some_field, list):
                n_graphs = len(some_field)
            elif isinstance(some_field, np.ndarray):
                n_graphs = some_field.shape[0]
            else:
                raise TypeError(
                    f"Unsupported type for field array in v7.3 graphs: {type(some_field)}"
                )

            graphs = []
            for i in range(n_graphs):
                d = {}
                for k, v in graphs_raw.items():
                    d[k] = v[i]
                graphs.append(SimpleNamespace(**d))

        # Case 2: already a list of dicts/objects
        elif isinstance(graphs_raw, list):
            graphs = [
                SimpleNamespace(**g) if isinstance(g, dict) else g
                for g in graphs_raw
            ]

        # Fallback: single struct-like object
        else:
            graphs = [graphs_raw]

    # ------------- check + convert globals to NumPy -------------
    required_names = ["A", "E", "G", "Iy", "Iz", "J", "Lx_ref", "elems", "bc"]

    for name in required_names:
        if globals_raw.get(name) is None:
            raise ValueError(f"Global '{name}' could not be loaded from MAT file.")
        globals_raw[name] = np.asarray(globals_raw[name])

    # unpack if you still like explicit names
    A_global = globals_raw["A"]
    E_global = globals_raw["E"]
    G_global = globals_raw["G"]
    Iy_global = globals_raw["Iy"]
    Iz_global = globals_raw["Iz"]
    J_global = globals_raw["J"]
    Lx_ref_global = globals_raw["Lx_ref"]
    elems_global = globals_raw["elems"]  # NEW
    bc_global = globals_raw["bc"]  # NEW
    bc_global = bc_global.astype(np.int64)

    # ------------------ clip to requested total_num_data ------------------
    n_available = len(graphs)
    if total_num_data > n_available:
        print(f"[generate_graphspaceframe_data_loader] "
              f"Warning: requested {total_num_data} samples, "
              f"but MAT file only has {n_available}; truncating.")
        total_num_data = n_available

    graphs = graphs[:total_num_data]

    data_list = []
    for g in graphs:
        # -------- read fields (NumPy first) --------
        pos_np   = np.asarray(g.nodes)                      # (N, d)
        elems_np = np.asarray(elems_global, dtype=np.int64)  # (M, 2), 1-based
        loads_np = np.asarray(getattr(g, 'loads', np.zeros_like(pos_np)))
        bc_mask_np = np.asarray(bc_global)

        target = getattr(g, 'target', None)
        if target is None:
            raise ValueError("Missing 'target' in a case; Option A expects 'target' per graph.")
        y = torch.tensor(np.asarray(target), dtype=torch.float32)

        # -------- torch tensors --------
        pos = torch.tensor(pos_np, dtype=torch.float32)     # (N,d)
        A_phys = torch.tensor(A_global, dtype=torch.float32)  # (M,1)
        N, d = pos.shape

        # Edge index (0-based), then make undirected
        edge_index_phys = torch.from_numpy(elems_np - 1).t().contiguous()  # (2, M)

        # -------- E from global E (scalar or array -> scalar for scaling) --------
        A_char = scalar_from_global(A_global, name="A")
        E_char = scalar_from_global(E_global, name="E")
        G_char = scalar_from_global(G_global, name="Iy")
        Iy_char = scalar_from_global(Iy_global, name="Iy")
        Iz_char = scalar_from_global(Iz_global, name="Iy")
        J_char = scalar_from_global(J_global, name="Iy")
        Lx_ref_val = scalar_from_global(Lx_ref_global, name="Lx_ref")

        # to_undirected and duplicate A accordingly
        edge_index_gnn = to_undirected(
            edge_index_phys, num_nodes=N
        )

        # -------- geometry-derived edge features --------
        L, rel = edge_lengths(pos, edge_index_gnn)
        cs = rel / L.clamp_min(1e-9)  # [E,3] = (cosx, cosy, cosz)

        # typical physical length from physical edges
        L_phys, _ = edge_lengths(pos, edge_index_phys)
        L_char = L_phys.mean().clamp_min(1e-12)

        # -------- normalize node coords --------

        x_left = pos[:, 0].min()
        y_bottom = pos[:, 1].min()
        z_bottom = pos[:, 2].min()
        pos_norm = torch.stack([(pos[:, 0] - x_left) / Lx_ref_val,
                                (pos[:, 1] - y_bottom) / Lx_ref_val,
                                (pos[:, 2] - z_bottom) / Lx_ref_val],dim=1)

        # -------- loads & scaling --------
        loads = torch.tensor(loads_np, dtype=torch.float32)   # (N,d)
        Fz = loads[:, 2] if loads.shape[1] >= 2 else torch.zeros(N, dtype=torch.float32)
        F_char = Fz.abs().max()
        if not torch.isfinite(F_char) or F_char.item() == 0.0:
            F_char = torch.tensor(1.0, dtype=torch.float32)

        Fz_scaled = (Fz / F_char).unsqueeze(1)  # (N,1)
        # if you actually want scaled Fy as feature, use Fy_scaled here:
        x_node = torch.cat([pos_norm, Fz_scaled], dim=1).float()  # (N,4)

        # choose your edge_attr; here: just direction cosines and length
        edge_attr_gnn = torch.cat([cs, L / Lx_ref_val], dim=1)

        # -------- target scaling --------

        s_u = (E_char * A_char) / (F_char * L_char)  # for axial displacement u
        s_v = (E_char * Iz_char) / (F_char * (L_char ** 3))  # for transverse displacement v
        s_w = (E_char * Iy_char) / (F_char * (L_char ** 3))  # for transverse displacement w
        s_thy = (E_char * Iy_char) / (F_char * (L_char ** 2))  # for rotation theta
        s_thz = (E_char * Iz_char) / (F_char * (L_char ** 2))  # for rotation theta
        s_thx = (G_char * J_char) / (F_char * (L_char ** 2))  # for rotation theta

        # S_y = torch.stack([s_u, s_v, s_w, s_thx, s_thy, s_thz], dim=-1)  # (6,)
        # if S_y.ndim == 1:
        #     S_y = S_y.view(1, 6)

        s_thx_eff = s_thx / L_char
        s_thy_eff = s_thy / L_char
        s_thz_eff = s_thz / L_char

        seq = (s_u * s_v * s_w * s_thx_eff * s_thy_eff * s_thz_eff) ** (1.0 / 6.0)

        S_y = torch.stack([seq, seq, seq, seq * L_char, seq * L_char, seq * L_char], dim=-1)  # (6,)
        S_y = S_y.to(y.dtype).to(y.device)
        if S_y.ndim == 1:
            S_y = S_y.view(1, 6)

        # kax = (E_char * A_char) / L_char  # axial stiffness
        # kbz = (12 * E_char * Iz_char) / L_char ** 3  # bending about z
        # kby = (12 * E_char * Iy_char) / L_char ** 3  # bending about y
        # kt = (G_char * J_char) / L_char  # torsion stiffness
        #
        # keq = np.sqrt(kax ** 2 + kbz ** 2 + kby ** 2 + kt ** 2)
        #
        # S_y = keq / F_char  # dimensionless scale

        y_scaled = y * S_y  # broadcast over nodes (assuming y is (N,6))

        # -------- build Data --------
        data = Data(
            x=x_node,
            pos=pos_norm,
            pos_raw=pos,
            edge_index=edge_index_gnn,
            edge_attr=edge_attr_gnn,
            y=y_scaled,
        )

        data.edge_index_phys = edge_index_phys
        data.A_phys = A_phys
        data.loads = loads

        bc_mask = torch.tensor(bc_mask_np, dtype=torch.float32)
        try:
            data.fixed = bc_mask.bool()
        except NameError:
            data.fixed = torch.zeros(N, d)

        data.f_char = F_char.view(1)
        data.A_char = A_char
        data.Iy_char = Iy_char
        data.Iz_char = Iz_char
        data.J_char = J_char
        data.scale_y = S_y

        data_list.append(data)

    # ------------------ splits & loaders ------------------
    train_ds = data_list[:n_train]
    val_ds = data_list[n_train:n_train + n_val]
    test_ds = data_list[n_train + n_val:]

    bs = int(config['train']['batchsize'])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs)
    test_loader  = DataLoader(test_ds,  batch_size=1)

    return train_loader, val_loader, test_loader


def scalar_from_global(x, name="value"):
    """
    Convert a scalar or array-like 'x' into a single torch.float32 scalar tensor.
    - If x is empty → error.
    - If scalar → convert directly.
    - If array → take mean() then convert.
    """
    x_np = np.asarray(x)

    if x_np.size == 0:
        raise ValueError(f"Global {name} is empty.")

    if x_np.ndim == 0:
        val = float(x_np)
    else:
        val = float(np.mean(x_np))

    return torch.tensor(val, dtype=torch.float32)

# ------------------ small helper for globals ------------------
def load_global_field(mat_dict, key, *, stack=False):
    """
    Load a global field from a MATLAB dict.

    - If stack=True: assumes mat_dict[key][0] is a list/1D array and stacks it.
    - Else: takes mat_dict[key][0] directly.
    - If resulting value has size 1 -> convert to float scalar.

    Returns: Python float or numpy array.
    """
    if key not in mat_dict:
        raise ValueError(f"Global '{key}' not found in MAT file.")

    raw = mat_dict[key]

    if stack:
        val = np.stack(raw)
    else:
        val = raw

    # scalar → float
    if np.size(val) == 1:
        val = float(np.squeeze(val))

    return val