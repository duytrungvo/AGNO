import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import math

from .utils_graph import _unscale_y_planeframe, rel_l2_stats_batch, _unscale_frame, save_epoch_attention_mat
from scipy.io import savemat


def val(model, loader, device, args, num_nodes_list):

    # extract the information of node numbers
    max_pde_nodes, max_bc_nodes, max_par_nodes = num_nodes_list

    all_errs_scaled = []
    all_errs_phys = []

    with torch.no_grad():
        for (par, coors, u, v, theta, flag, par_flag, S_y) in loader:

            par = par.float().to(device)
            par_flag = par_flag.float().to(device)
            coors = coors.float().to(device)
            u = u.float().to(device)
            v = v.float().to(device)
            theta = theta.float().to(device)
            S_y = S_y.float().to(device)

            # extract shape coordinates
            if args.geo_node == 'vary_bound' or 'vary_bound_sup':
                ss_index = np.arange(max_pde_nodes, max_pde_nodes + max_bc_nodes)
            if args.geo_node == 'all_domain':
                ss_index = np.arange(0, max_pde_nodes + max_bc_nodes)
            shape_coor = coors[:, ss_index, :].float().to(device)    # (B, max_bcxy, 2)
            shape_flag = flag[:, ss_index]
            shape_flag = shape_flag.float().to(device)    # (B, max_bcxy)

            # model forward
            u_pred, v_pred, theta_pred = model(coors[:,:,0], coors[:,:,1], par, par_flag, shape_coor, shape_flag)

            # get the flag
            flag_valid = torch.where(flag>=0, torch.ones_like(flag), torch.zeros_like(flag)).float().to(device)

            # ---------- 1) RELATIVE L2 IN SCALED SPACE (as before) ----------
            num_scaled = torch.sum(
                (u_pred * flag_valid - u * flag_valid) ** 2
                + (v_pred * flag_valid - v * flag_valid) ** 2
                + (theta_pred * flag_valid - theta * flag_valid) ** 2,
                dim=-1
            )

            den_scaled = torch.sum(
                (u * flag_valid) ** 2
                + (v * flag_valid) ** 2
                + (theta * flag_valid) ** 2,
                dim=-1
            ) + 1e-16

            L2_relative_scaled = torch.sqrt(num_scaled / den_scaled)  # (B,)
            all_errs_scaled.append(L2_relative_scaled.detach().cpu().numpy())

            # ---------- 2) RELATIVE L2 IN PHYSICAL SPACE (after unscale) ----------
            # S_y columns: [s_u, s_v, s_theta]
            s_u = S_y[:, 0].view(-1, 1)  # (B,1)
            s_v = S_y[:, 1].view(-1, 1)
            s_theta = S_y[:, 2].view(-1, 1)

            # NOTE: assuming scaled = physical * S_y  →  physical = scaled / S_y
            # If your implementation instead used scaled = physical / S_y,
            # then change all "/" below to "*".
            u_pred_phys = u_pred / s_u
            v_pred_phys = v_pred / s_v
            theta_pred_phys = theta_pred / s_theta

            u_phys = u / s_u
            v_phys = v / s_v
            theta_phys = theta / s_theta

            num_phys = torch.sum(
                (u_pred_phys * flag_valid - u_phys * flag_valid) ** 2
                + (v_pred_phys * flag_valid - v_phys * flag_valid) ** 2
                + (theta_pred_phys * flag_valid - theta_phys * flag_valid) ** 2,
                dim=-1
            )

            den_phys = torch.sum(
                (u_phys * flag_valid) ** 2
                + (v_phys * flag_valid) ** 2
                + (theta_phys * flag_valid) ** 2,
                dim=-1
            ) + 1e-16

            L2_relative_phys = torch.sqrt(num_phys / den_phys)  # (B,)
            all_errs_phys.append(L2_relative_phys.detach().cpu().numpy())

    # stack all batches → (N_val,)
    all_errs_scaled = np.concatenate(all_errs_scaled)
    all_errs_phys = np.concatenate(all_errs_phys)

    mean_scaled = all_errs_scaled.mean()
    std_scaled = all_errs_scaled.std(ddof=1)

    mean_phys = all_errs_phys.mean()
    std_phys = all_errs_phys.std(ddof=1)

    return mean_scaled, std_scaled, mean_phys, std_phys


def test(model, loader, device, args, num_nodes_list):

    # extract the information of node numbers
    max_pde_nodes, max_bc_nodes, max_par_nodes = num_nodes_list

    test_err_scaled = []
    test_err_phys = []

    max_relative_err = -1
    min_relative_err = np.inf
    test_in_coor = []
    test_in_par = []
    test_out = []
    pred = []

    with torch.no_grad():
        for (par, coors, u, v, theta, flag, par_flag, S_y) in loader:
            par = par.float().to(device)
            par_flag = par_flag.float().to(device)
            coors = coors.float().to(device)
            u = u.float().to(device)
            v = v.float().to(device)
            theta = theta.float().to(device)
            flag = flag.float().to(device)
            S_y = S_y.float().to(device)

            # extract shape coordinates
            if args.geo_node == 'vary_bound' or 'vary_bound_sup':
                ss_index = np.arange(max_pde_nodes, max_pde_nodes + max_bc_nodes)
            if args.geo_node == 'all_domain':
                ss_index = np.arange(0, max_pde_nodes + max_bc_nodes)

            shape_coor = coors[:, ss_index, :].float().to(device)    # (B, max_bcxy, 2)
            shape_flag = flag[:, ss_index]
            shape_flag = shape_flag.float().to(device)    # (B, max_bcxy)

            # model forward
            u_pred, v_pred, theta_pred = model(coors[:,:,0], coors[:,:,1], par, par_flag, shape_coor, shape_flag)

            # save sample for .mat (still scaled, as in your original code)
            test_in_coor.append(coors[0].cpu())
            test_in_par.append(par[0].cpu())
            test_out.append(np.vstack([
                u[0].detach().cpu().numpy(),
                v[0].detach().cpu().numpy(),
                theta[0].detach().cpu().numpy()
            ]).T)
            pred.append(np.vstack([
                u_pred[0].detach().cpu().numpy(),
                v_pred[0].detach().cpu().numpy(),
                theta_pred[0].detach().cpu().numpy()
            ]).T)

            # get the flag
            flag_valid = torch.where(flag>=0, torch.ones_like(flag), torch.zeros_like(flag)).float().to(device)

            # ---------- 1) RELATIVE L2 IN SCALED SPACE (as before) ----------
            num_scaled = torch.sum(
                (u_pred * flag_valid - u * flag_valid) ** 2
                + (v_pred * flag_valid - v * flag_valid) ** 2
                + (theta_pred * flag_valid - theta * flag_valid) ** 2,
                dim=-1
            )

            den_scaled = torch.sum(
                (u * flag_valid) ** 2
                + (v * flag_valid) ** 2
                + (theta * flag_valid) ** 2,
                dim=-1
            ) + 1e-16

            L2_relative_scaled = torch.sqrt(num_scaled / den_scaled)  # (B,)
            test_err_scaled.append(L2_relative_scaled.detach().cpu().numpy())

            # ---------- 2) RELATIVE L2 IN PHYSICAL SPACE (after unscale) ----------
            # S_y columns: [s_u, s_v, s_theta]
            s_u = S_y[:, 0].view(-1, 1)  # (B,1)
            s_v = S_y[:, 1].view(-1, 1)
            s_theta = S_y[:, 2].view(-1, 1)

            # NOTE: assuming scaled = physical * S_y  →  physical = scaled / S_y
            # If your implementation instead used scaled = physical / S_y,
            # then change all "/" below to "*".
            u_pred_phys = u_pred / s_u
            v_pred_phys = v_pred / s_v
            theta_pred_phys = theta_pred / s_theta

            u_phys = u / s_u
            v_phys = v / s_v
            theta_phys = theta / s_theta

            num_phys = torch.sum(
                (u_pred_phys * flag_valid - u_phys * flag_valid) ** 2
                + (v_pred_phys * flag_valid - v_phys * flag_valid) ** 2
                + (theta_pred_phys * flag_valid - theta_phys * flag_valid) ** 2,
                dim=-1
            )

            den_phys = torch.sum(
                (u_phys * flag_valid) ** 2
                + (v_phys * flag_valid) ** 2
                + (theta_phys * flag_valid) ** 2,
                dim=-1
            ) + 1e-16

            L2_relative_phys = torch.sqrt(num_phys / den_phys)  # (B,)
            test_err_phys.append(L2_relative_phys.detach().cpu().numpy())

            # find the max and min error sample in this batch
            max_err, max_err_idx = torch.topk(L2_relative_phys, 1)
            if max_err > max_relative_err:
                max_relative_err = max_err
                worst_xcoor = coors[max_err_idx,:,0].squeeze(0).squeeze(-1).detach().cpu().numpy()
                worst_ycoor = coors[max_err_idx,:,1].squeeze(0).squeeze(-1).detach().cpu().numpy()
                worst_f = u_pred[max_err_idx,:].squeeze(0).detach().cpu().numpy()
                worst_gt = u[max_err_idx,:].squeeze(0).detach().cpu().numpy()
                worst_v = v_pred[max_err_idx, :].squeeze(0).detach().cpu().numpy()
                worst_v_gt = v[max_err_idx, :].squeeze(0).detach().cpu().numpy()
                worst_ff = flag[max_err_idx,:].squeeze(0).detach().cpu().numpy()
                valid_id = np.where(worst_ff>=-0.1)[0]
                worst_xcoor = worst_xcoor[valid_id]
                worst_ycoor = worst_ycoor[valid_id]
                worst_f = worst_f[valid_id]
                worst_gt = worst_gt[valid_id]
                worst_ff = worst_ff[valid_id]
            min_err, min_err_idx = torch.topk(-L2_relative_phys, 1)
            min_err = -min_err
            if min_err < min_relative_err:
                min_relative_err = min_err
                best_xcoor = coors[min_err_idx,:,0].squeeze(0).squeeze(-1).detach().cpu().numpy()
                best_ycoor = coors[min_err_idx,:,1].squeeze(0).squeeze(-1).detach().cpu().numpy()
                best_f = u_pred[min_err_idx,:].squeeze(0).detach().cpu().numpy()
                best_gt = u[min_err_idx,:].squeeze(0).detach().cpu().numpy()
                best_ff = flag[min_err_idx,:].squeeze(0).detach().cpu().numpy()
                valid_id = np.where(best_ff>=-0.1)[0]
                best_xcoor = best_xcoor[valid_id]
                best_ycoor = best_ycoor[valid_id]
                best_f = best_f[valid_id]
                best_gt = best_gt[valid_id]
                best_ff = best_ff[valid_id]

    # concatenate errors
    test_err_scaled = np.concatenate(test_err_scaled)  # (N_samples,)
    test_err_phys = np.concatenate(test_err_phys)  # (N_samples,)

    mean_err_scaled = test_err_scaled.mean()
    std_err_scaled = test_err_scaled.std(ddof=1)

    mean_err_phys = test_err_phys.mean()
    std_err_phys = test_err_phys.std(ddof=1)

    savemat(r'./res/saved_models/{}_test_{}.mat'.format(args.data, args.model),
            {'x': test_in_coor, 'par': test_in_par, 'yp': pred, 'y': test_out})

    # color bar range
    max_color = np.amax([np.amax(worst_gt), np.amax(best_gt)])
    min_color = np.amin([np.amin(worst_gt), np.amin(best_gt)])

    # make the plots
    cm = plt.cm.get_cmap('RdYlBu')
    plt.figure(figsize=(15,8))
    plt.subplot(2,3,1)
    plt.scatter(worst_xcoor, worst_ycoor, c=worst_f, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=5)
    plt.colorbar()
    plt.title('prediction')
    plt.subplot(2,3,2)
    plt.scatter(worst_xcoor, worst_ycoor, c=worst_gt, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=5)
    plt.title('ground truth')
    plt.colorbar()
    plt.subplot(2,3,3)
    plt.scatter(worst_xcoor, worst_ycoor, c=np.abs(worst_f-worst_gt), cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=5)
    plt.title('absolute error')
    plt.colorbar()
    plt.subplot(2,3,4)
    plt.scatter(best_xcoor, best_ycoor, c=best_f, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=5)
    plt.colorbar()
    plt.title('prediction')
    plt.subplot(2,3,5)
    plt.scatter(best_xcoor, best_ycoor, c=best_gt, cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=5)
    plt.title('ground truth')
    plt.colorbar()
    plt.subplot(2,3,6)
    plt.scatter(best_xcoor, best_ycoor, c=np.abs(best_f-best_gt), cmap=cm, vmin=min_color, vmax=max_color, marker='o', s=5)
    plt.title('absolute error')
    plt.colorbar()
    plt.savefig(r'./res/plots/sample_{}_{}_{}.png'.format(args.geo_node, args.model, args.data))

    # plt.figure(figsize=(15, 8), dpi=400)
    # iid = np.arange(0, worst_xcoor.shape[0], 1)
    # plt.scatter(worst_xcoor[iid] + worst_gt[iid], worst_ycoor[iid] + worst_v_gt[iid], facecolors='none',
    #             edgecolors='b', marker='o', s=50, label='Ground Truth')
    # plt.scatter(worst_xcoor[iid] + worst_f[iid], worst_ycoor[iid] + worst_v[iid],
    #             color='r', marker='x', s=10, label='GANO')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend(frameon=False)
    # # plt.show()
    # plt.savefig(r'./res/plots/sample2_{}_{}_{}.png'.format(args.geo_node, args.model, args.data))

    return mean_err_scaled, std_err_scaled, mean_err_phys, std_err_phys


def sup_train(args, config, model, device, loaders, num_nodes_list):
    # print training configuration
    print('training configuration')
    print('batchsize:', config['train']['batchsize'])
    print('coordinate sampling frequency:', config['train']['coor_sampling_freq'])
    print('learning rate:', config['train']['base_lr'])

    # get train and test loader
    train_loader, val_loader, test_loader = loaders

    # get number of nodes of different type
    max_pde_nodes, max_bc_nodes, max_par_nodes = num_nodes_list

    # define model training configuration
    pbar = range(config['train']['epochs'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    # define optimizer and loss
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['base_lr'])

    # visual frequency
    vf = config['train']['visual_freq']

    # move the model to the defined device
    try:
        model.load_state_dict(
            torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model),
                       map_location=device))
    except:
        print('No trained models')
    model = model.to(device)

    # start the training
    if args.phase == 'train':
        min_val_err = np.inf
        avg_pde_loss = np.inf
        avg_bc_loss = np.inf
        eval_err = []
        for e in pbar:

            # show the performance improvement
            if e % vf == 0:
                model.eval()
                err, std, err_phys, std_phys = val(model, val_loader, device, args, num_nodes_list)
                print(
                    f"Epoch error: scaled = {err:.18f} (std {std:.18f}),  "
                    f"physical = {err_phys:.18f} (std {std_phys:.18f})")

                print('current epochs pde loss:', avg_pde_loss, 'bc loss:', avg_bc_loss)

                if err_phys < min_val_err:
                    torch.save(model.state_dict(),
                               r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data,
                                                                                    args.model))
                    min_val_err = err_phys

            avg_pde_loss = 0
            avg_bc_loss = 0
            # train one epoch
            model.train()
            for (par, coors, u, v, theta, flag, par_flag, _) in train_loader:

                for _ in range(config['train']['coor_sampling_freq']):

                    # extract bc condition coordinates
                    all_coors = coors[:, :, :].float().to(device)
                    all_flag = flag[:, :]
                    all_flag = torch.where(all_flag > -0.5, torch.ones_like(all_flag),
                                           torch.zeros_like(all_flag)).float().to(device)
                    u_gt = u[:, :].float().to(device)
                    v_gt = v[:, :].float().to(device)
                    theta_gt = theta[:, :].float().to(device)

                    # prepare the parameter input
                    par = par.float().to(device)
                    par_flag = par_flag.float().to(device)

                    # prepare the shape coordinate input
                    if args.geo_node == 'vary_bound_sup':
                        ss_index = np.arange(max_pde_nodes, max_pde_nodes + max_bc_nodes)
                    if args.geo_node == 'all_domain':
                        ss_index = np.arange(0, max_pde_nodes + max_bc_nodes)
                    shape_coor = coors[:, ss_index, :].float().to(device)  # (B, max_bcxy, 2)
                    shape_flag = flag[:, ss_index]
                    shape_flag = shape_flag.float().to(device)  # (B, max_bcxy)

                    # forward to get the prediction on fixed boundary
                    u_pred, v_pred, theta_pred = model(all_coors[:, :, 0], all_coors[:, :, 1], par, par_flag,
                                                       shape_coor, shape_flag)

                    # compute the losses
                    total_loss = mse(u_pred * all_flag, u_gt * all_flag) \
                                 + mse(v_pred * all_flag, v_gt * all_flag) \
                                 + mse(theta_pred * all_flag, theta_gt * all_flag)

                    # store the loss
                    avg_pde_loss += total_loss.detach().cpu().item()

                    # update parameter
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

            eval_err.append([e, avg_pde_loss, avg_bc_loss, err_phys, std_phys])
            err, std, err_phys, std_phys = float('nan'), float('nan'), float('nan'), float('nan')

        # save evaluation err to text file
        np.savetxt(r'./res/saved_models/evaluation_error_{}_{}.txt'.format(args.data, args.model), np.array(eval_err),
                   fmt='%.6e')
    # final test
    model.load_state_dict(
        torch.load(r'./res/saved_models/best_model_{}_{}_{}.pkl'.format(args.geo_node, args.data, args.model),
                   map_location=device))
    model.eval()
    # save memory for the test
    del train_loader, val_loader
    mean_err, std_err, mean_err_phys, std_err_phys = test(model, test_loader, device, args, num_nodes_list)
    print('Best L2 relative error on test loader:', mean_err_phys, 'and std ', std_err_phys)


def graph_val(args, config, model, device, loaders):
    # get train and test loader
    train_loader, val_loader, test_loader = loaders

    # -------- VAL --------
    va_rel_sum, va_rel_sumsq, va_rel_cnt = 0.0, 0.0, 0
    va_rel_phys_sum, va_rel_phys_sumsq, va_rel_phys_cnt = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch)

            fixed_mask = batch.fixed.bool() if hasattr(batch, "fixed") else None

            # 1) scaled-space metric
            s, ssq, c = rel_l2_stats_batch(pred, batch.y, fixed_mask, batch.batch)
            va_rel_sum += s
            va_rel_sumsq += ssq
            va_rel_cnt += c

            # 2) physical-space metric (like test)
            pred_phys, y_phys = _unscale_frame(pred, batch.y, batch.scale_y, batch.batch)
            s_p, ssq_p, c_p = rel_l2_stats_batch(pred_phys, y_phys, fixed_mask, batch.batch)
            va_rel_phys_sum += s_p
            va_rel_phys_sumsq += ssq_p
            va_rel_phys_cnt += c_p

    va_rel2 = va_rel_sum / max(va_rel_cnt, 1)
    va_rel2_phys = va_rel_phys_sum / max(va_rel_phys_cnt, 1)

    # Std of per-graph relative L2 (unbiased/Bessel if cnt>1; else NaN)
    if va_rel_cnt > 1:
        var = max(va_rel_sumsq - (va_rel_sum * va_rel_sum) / va_rel_cnt, 0.0) / (va_rel_cnt - 1)
        va_rel_std = math.sqrt(var)
    else:
        va_rel_std = float('nan')

    if va_rel_phys_cnt > 1:
        var_phys = max(va_rel_phys_sumsq - (va_rel_phys_sum * va_rel_phys_sum) / va_rel_phys_cnt, 0.0) \
                   / (va_rel_phys_cnt - 1)
        va_rel_phys_std = math.sqrt(var_phys)
    else:
        va_rel_phys_std = float('nan')

    return va_rel2, va_rel_std, va_rel2_phys, va_rel_phys_std


def graph_test(args, config, model, device, loaders):
    _, _, test_loader = loaders

    graphs = []

    va_rel_sum, va_rel_sumsq, va_rel_cnt = 0.0, 0.0, 0

    # ensure save dir exists
    # save_dir = './res/saved_models'
    # os.makedirs(save_dir, exist_ok=True)
    # save_file = os.path.join(save_dir, f'{args.data}_test.mat')

    with torch.no_grad():
        for batch in test_loader:

            batch = batch.to(device)
            pred_scaled = model(batch)          # [N, d]
            y_scaled    = batch.y               # [N, d]

            # Unscale robustly for any batch size
            y_pred, y_true = _unscale_frame(pred_scaled, y_scaled, batch.scale_y, batch.batch)

            # ---- Per-graph split (works even if batch_size>1) ----
            G = int(batch.batch.max().item()) + 1 if batch.batch.numel() > 0 else 1

            # total nodes in this Batch
            N_total = batch.pos_raw.size(0)  # or batch.x.size(0)

            for g in range(G):
                if G > 1:
                    node_idx = torch.nonzero(batch.batch == g, as_tuple=False).view(-1)  # [Ng]
                    if node_idx.numel() == 0:
                        continue
                else:
                    # single-graph batch: take all nodes [0..N_total-1]
                    node_idx = torch.arange(N_total, device=device)

                # --- slice node-wise tensors using LongIndex ---
                pos_raw_g = batch.pos_raw.index_select(0, node_idx).detach().cpu().numpy()  # (Ng,2)
                loads_g = batch.loads.index_select(0, node_idx).detach().cpu().numpy()  # (Ng,d)
                y_pred_g = y_pred.index_select(0, node_idx).detach().cpu().numpy()  # (Ng,d)
                y_true_g = y_true.index_select(0, node_idx).detach().cpu().numpy()  # (Ng,d)

                # --- per-graph scalars (be robust to shapes [] vs [G]) ---
                if hasattr(batch, 'f_char'):
                    f_char_tensor = batch.f_char
                    if torch.is_tensor(f_char_tensor) and f_char_tensor.ndim > 0:
                        f_char_g = f_char_tensor[g].view(1).detach().cpu().numpy()
                    else:
                        f_char_g = torch.as_tensor(f_char_tensor).view(1).cpu().numpy()
                else:
                    f_char_g = np.array([1.0], dtype=np.float32)

                if hasattr(batch, 'A_char'):
                    A_char_tensor = batch.A_char
                    if torch.is_tensor(A_char_tensor) and A_char_tensor.ndim > 0:
                        A_char_g = A_char_tensor[g].view(1).detach().cpu().numpy()
                    else:
                        A_char_g = torch.as_tensor(A_char_tensor).view(1).cpu().numpy()
                else:
                    A_char_g = np.array([1.0], dtype=np.float32)

                # --- Build subgraph edges for graph g ---
                # map global node id -> local [0..Ng-1]
                inv = -torch.ones(N_total, dtype=torch.long, device=device)
                inv[node_idx] = torch.arange(node_idx.numel(), device=device)

                e = batch.edge_index_phys  # [2, M_all], on same device
                M_all = batch.edge_index_phys.size(1)
                keep = (inv[e[0]] >= 0) & (inv[e[1]] >= 0)  # [M_all]
                e_local = torch.stack([inv[e[0, keep]], inv[e[1, keep]]], dim=0)  # [2, Mg]

                elems_matlab_g = (e_local.t().contiguous().detach().cpu().numpy() + 1).astype(np.int64)

                # ---- A_phys: either scalar per graph [G] or vector per element [M_all] ----
                if hasattr(batch, "A_phys"):
                    A_attr = batch.A_phys

                    # ensure tensor, device, dtype
                    if not torch.is_tensor(A_attr):
                        A_attr = torch.as_tensor(A_attr, device=device, dtype=torch.float32)
                    else:
                        A_attr = A_attr.to(device=device, dtype=torch.float32)

                    # If stored as (M,1) somewhere, squeeze to (M,)
                    if A_attr.ndim == 2 and A_attr.size(1) == 1:
                        A_attr = A_attr.view(-1)

                    # Case 1: scalar per graph -> shape [G]
                    if A_attr.ndim == 1 and A_attr.numel() == G:
                        # one scalar for this graph g, store as length-1 numpy array
                        A_phys_g = A_attr[g].view(1).detach().cpu().numpy().astype(np.float32)  # (1,)

                    # Case 2: vector per element across batch -> shape [M_all]
                    elif A_attr.ndim == 1 and A_attr.numel() == M_all:
                        # one value per physical element; recover this graph's vector via keep
                        A_phys_g = A_attr[keep].detach().cpu().numpy().astype(np.float32)  # (Mg,)

                    else:
                        raise ValueError(
                            f"Unexpected A_phys shape {tuple(A_attr.shape)}; "
                            f"expected [G={G}] (scalar per graph) or [M_all={M_all}] (vector per element)."
                        )
                else:
                    # Fallback: no A_phys present
                    A_phys_g = np.array([1.0], dtype=np.float32)

                graphs.append({
                    'nodes': pos_raw_g,  # [Ng,2]
                    'elements': elems_matlab_g,  # [Mg,2] (1-based)
                    'A': A_phys_g,  # [Mg,1]
                    'loads': loads_g,  # [Ng,d]
                    'y_true': y_true_g,  # [Ng,d]
                    'y_pred': y_pred_g,  # [Ng,d]
                    'F_char': f_char_g,  # [1]
                    'A_char': A_char_g,  # [1]
                })

            # ---- Metrics on physical (unscaled) values ----
            fixed_mask = batch.fixed.bool() if hasattr(batch, "fixed") else None
            s, ssq, c = rel_l2_stats_batch(y_pred, y_true, fixed_mask, batch.batch)
            va_rel_sum    += s
            va_rel_sumsq  += ssq
            va_rel_cnt    += c

    # savemat(save_file, {'graphs': np.array(graphs, dtype=object)})
    savemat(r'./res/saved_models/{}_test_{}.mat'.format(args.data, args.model), {'graphs': np.array(graphs, dtype=object)})

    va_rel2 = va_rel_sum / max(va_rel_cnt, 1)
    if va_rel_cnt > 1:
        var = max(va_rel_sumsq - (va_rel_sum * va_rel_sum) / va_rel_cnt, 0.0) / (va_rel_cnt - 1)
        va_rel_std = math.sqrt(var)
    else:
        va_rel_std = float('nan')

    return va_rel2, va_rel_std


def sup_graph_train(args, config, model, device, loaders):

    # print training configuration
    print('training configuration')
    print('batchsize:', config['train']['batchsize'])
    print('learning rate:', config['train']['base_lr'])

    # get train and test loader
    train_loader, val_loader, test_loader = loaders
    # training setup
    mse = nn.MSELoss()  # instead of nn.L1Loss()
    # (optional) reduce LR slightly for stability with MSE
    opt = optim.Adam(model.parameters(), lr=config['train']['base_lr'])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])

    vf = config['train']['visual_freq']

    epoch_bar = tqdm(range(config['train']['epochs']), dynamic_ncols=True, smoothing=0.1)

    # move the model to the defined device
    try:
        model.load_state_dict(
            torch.load(r'./res/saved_models/best_model_{}_{}.pkl'.format(args.data, args.model),
                       map_location=device))
    except:
        print('No trained models')

    model = model.to(device)
    # start the training
    if args.phase == 'train':
        min_val_err = np.inf
        avg_train_loss = np.inf
        avg_bc_loss = np.inf
        eval_err = []
        bad = 0
        patience = config['train']['patience']

        for e in epoch_bar:

            # show the performance improvement
            if e % vf == 0:
                model.eval()
                err_scaled, std_scaled, err_phys, std_phys = graph_val(args, config, model, device, loaders)

                print(
                    f"Epoch error: scaled = {err_scaled:.18f} (std {std_scaled:.18f}),  "
                    f"physical = {err_phys:.18f} (std {std_phys:.18f})")
                print('current epochs train loss:', avg_train_loss, 'bc loss:', avg_bc_loss)

                if err_phys < min_val_err:
                    torch.save(model.state_dict(),
                               r'./res/saved_models/best_model_{}_{}.pkl'.format(args.data, args.model))
                    min_val_err = err_phys
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break

            # -------- TRAIN --------
            model.train()
            avg_train_loss = 0.0
            avg_bc_loss = 0.0

            for batch in train_loader:
                batch = batch.to(device)
                y = batch.y         # [N,3]
                pred = model(batch)
                loss = (
                        mse(pred[:, 0:1], y[:, 0:1]) +
                        mse(pred[:, 1:2], y[:, 1:2]) +
                        mse(pred[:, 2:3], y[:, 2:3])
                )

                avg_train_loss += loss.item()

                # update parameter
                opt.zero_grad()
                loss.backward()
                opt.step()

            # scheduler + ckpt
            scheduler.step()

            eval_err.append([e, avg_train_loss, avg_bc_loss, err_phys, std_phys])
            err_scaled, std_scaled, err_phys, std_phys = float('nan'), float('nan'), float('nan'), float('nan')

        # save evaluation err to text file
        np.savetxt(r'./res/saved_models/evaluation_error_{}_{}.txt'.format(args.data, args.model), np.array(eval_err), fmt='%.6e')

    # # final test
    model.load_state_dict(
        torch.load(r'./res/saved_models/best_model_{}_{}.pkl'.format(args.data, args.model),
                   map_location=device))
    model.eval()
    # save memory for the test
    del train_loader, val_loader
    mean_err, std_err = graph_test(args, config, model, device, loaders)
    print('Best L2 relative error on test loader:', mean_err, 'and std ', std_err)


def sup_graph_attn_train(args, config, model, device, loaders):

    # print training configuration
    print('training configuration')
    print('batchsize:', config['train']['batchsize'])
    print('learning rate:', config['train']['base_lr'])

    # get train and test loader
    train_loader, val_loader, test_loader = loaders
    # training setup
    mse = nn.MSELoss()  # instead of nn.L1Loss()
    # (optional) reduce LR slightly for stability with MSE
    opt = optim.Adam(model.parameters(), lr=config['train']['base_lr'])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])

    vf = config['train']['visual_freq']

    epoch_bar = tqdm(range(config['train']['epochs']), dynamic_ncols=True, smoothing=0.1)

    # move the model to the defined device
    try:
        model.load_state_dict(
            torch.load(r'./res/saved_models/best_model_{}_{}.pkl'.format(args.data, args.model),
                       map_location=device))
    except:
        print('No trained models')

    model = model.to(device)
    # start the training
    if args.phase == 'train':
        min_val_err = np.inf
        avg_train_loss = np.inf
        avg_bc_loss = np.inf
        eval_err = []
        bad = 0
        patience = config['train']['patience']

        # ---------------- Save epoch 0 attention (before any training) ----------------
        model.eval()
        epoch_attn0 = []
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(device)
                _, attn = model(batch, return_last_attn=True)
                epoch_attn0.append({
                    "alpha": attn["alpha"].detach().cpu(),
                    "batch": attn["batch"].detach().cpu()
                })
        save_epoch_attention_mat(args, 0, epoch_attn0, save_dir="./res/saved_models")

        for e in epoch_bar:

            # -------- VALIDATE (every vf epochs) --------
            err_scaled = std_scaled = err_phys = std_phys = float("nan")
            # completed epochs at start of loop == e
            validate_this_epoch = (e % vf == 0)
            if validate_this_epoch:
                model.eval()
                err_scaled, std_scaled, err_phys, std_phys = graph_val(args, config, model, device, loaders)

                print(
                    f"Epoch error: scaled = {err_scaled:.18f} (std {std_scaled:.18f}),  "
                    f"physical = {err_phys:.18f} (std {std_phys:.18f})")
                print('current epochs train loss:', avg_train_loss, 'bc loss:', avg_bc_loss)

                if err_phys < min_val_err:
                    torch.save(model.state_dict(),
                               r'./res/saved_models/best_model_{}_{}.pkl'.format(args.data, args.model))
                    min_val_err = err_phys
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break

            # -------- TRAIN --------
            model.train()
            avg_train_loss = 0.0
            avg_bc_loss = 0.0
            epoch_attn = []
            save_attn_after_train = ((e + 1) % (5*vf) == 0)  # completed epochs 100,200,300,...

            for batch in train_loader:
                batch = batch.to(device)
                y = batch.y         # [N,3]
                pred, attn = model(batch, return_last_attn=True)
                loss = (
                        mse(pred[:, 0:1], y[:, 0:1]) +
                        mse(pred[:, 1:2], y[:, 1:2]) +
                        mse(pred[:, 2:3], y[:, 2:3])
                )

                avg_train_loss += loss.item()

                # update parameter
                opt.zero_grad()
                loss.backward()
                opt.step()

                if save_attn_after_train:
                    epoch_attn.append({
                        "alpha": attn["alpha"].detach().cpu(),
                        "batch": attn["batch"].detach().cpu()
                    })

            # scheduler + ckpt
            scheduler.step()

            if save_attn_after_train:
                save_epoch_attention_mat(args, e+1, epoch_attn, save_dir="./res/saved_models")

            eval_err.append([e, avg_train_loss, avg_bc_loss, err_phys, std_phys])

        # save evaluation err to text file
        np.savetxt(r'./res/saved_models/evaluation_error_{}_{}.txt'.format(args.data, args.model), np.array(eval_err), fmt='%.6e')

    # # final test
    model.load_state_dict(
        torch.load(r'./res/saved_models/best_model_{}_{}.pkl'.format(args.data, args.model),
                   map_location=device))
    model.eval()
    # save memory for the test
    del train_loader, val_loader
    mean_err, std_err = graph_test(args, config, model, device, loaders)
    print('Best L2 relative error on test loader:', mean_err, 'and std ', std_err)

