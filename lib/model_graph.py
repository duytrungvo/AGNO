import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, scatter  # PyG >= 2.5

# ---------- small helpers ----------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=None, act=nn.GELU, dropout=0.0):
        super().__init__()
        hidden = hidden or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), act(),
            nn.Linear(hidden, out_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.net(x))


class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super().__init__()
        self.layers = nn.ModuleList()
        for j in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))
            if j != len(layers) - 2:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))
                self.layers.append(nonlinearity())
        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


# ---------- one message-passing block ----------
class MPNNBlock(MessagePassing):
    def __init__(self, h_dim, in_edg: int = 5, aggr='mean', dropout=0.0):
        super().__init__(aggr=aggr)
        # Edge embedding from physics-aware features: [A, 1/L, cosx, cosy, A/L] -> h_dim
        self.edge_mlp = MLP(in_dim=in_edg, out_dim=h_dim, hidden=h_dim, dropout=dropout)
        # Message and update
        self.msg_mlp  = MLP(h_dim + h_dim, h_dim, hidden=h_dim, dropout=dropout)  # [x_j || e_emb] -> h
        self.upd_mlp  = MLP(h_dim, h_dim, hidden=h_dim, dropout=dropout)
        self.norm     = nn.LayerNorm(h_dim)

    def forward(self, x, edge_index, edge_attr, pos_raw):
        feats = edge_attr
        e_emb = self.edge_mlp(feats)                                        # [E,h]
        return self.propagate(edge_index, x=x, e_emb=e_emb)

    def message(self, x_j, e_emb):
        # build message from neighbor state and edge embedding
        return self.msg_mlp(torch.cat([x_j, e_emb], dim=-1))

    def update(self, aggr_out, x):
        # residual + norm
        h = x + self.upd_mlp(aggr_out)
        return self.norm(h)


class GNOBlock(MessagePassing):
    """
    Graph Neural Operator block:
      m_{j->i} = K_ij(edge_attr) @ x_j
      x_i^{new} = x_i + sum_j m_{j->i}
    """
    def __init__(self, width, ker_in, ker_width=64, aggr='sum',
                 norm=True, symmetrize=False):
        """
        aggr:
          - 'sum'   : recommended for structural displacement prediction
          - 'mean'  : allowed, but usually less physical
        """
        super().__init__(aggr=aggr)

        self.width = width
        self.symmetrize = symmetrize

        # Kernel network: edge_attr -> (C x C) operator
        self.phi = DenseNet(
            layers=[ker_in, ker_width, ker_width, width * width],
            nonlinearity=nn.ReLU,
            normalize=False
        )

        self.ln = nn.LayerNorm(width) if norm else nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        """
        x:         [N, C]    node features (loads, BC mask, etc.)
        edge_attr: [E, ker_in]  geometry features (dir cosines, log(L), ...)
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        """
        Build dense operator kernel and apply to source node features.
        """
        E = edge_attr.size(0)
        C = self.width

        # edge_attr -> [E, C*C] -> [E, C, C]
        W = self.phi(edge_attr).view(E, C, C)

        if self.symmetrize:
            W = 0.5 * (W + W.transpose(-1, -2))

        # m_{j->i} = K_ij @ x_j
        mj = torch.bmm(x_j.unsqueeze(1), W).squeeze(1)  # [E, C]
        return mj

    def update(self, aggr_out, x):
        """
        Residual update + normalization.
        """
        return self.ln(x + aggr_out)


def apply_dirichlet(y_hat, fixed_mask, bc=None):
    """
    y_hat: [N,2] raw decoder output (in the same scale as y/targets)
    fixed_mask: [N,2] bool/float (1=True=fixed)
    bc: [N,2] prescribed displacement (default 0)
    Returns y_out that exactly satisfies u = bc on fixed DOFs.
    """
    if bc is None:
        bc = torch.zeros_like(y_hat)
    free = 1.0 - fixed_mask.float()
    return bc + free * (y_hat - bc)

# ---------- full model ----------
class GNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enc = MLP(config['model']['in_dim'],
                       config['model']['fc_dim'],
                       hidden=config['model']['fc_dim'],
                       dropout=config['model']['dropout'])
        self.blocks = nn.ModuleList([MPNNBlock(config['model']['fc_dim'],
                                               config['model']['edge_dim'],
                                               aggr='mean',
                                               dropout=config['model']['dropout'])
                                     for _ in range(config['model']['N_layer'])])
        self.dec = nn.Sequential(
            nn.GELU(),
            nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'] // 2),
            nn.GELU(),
            nn.Linear(config['model']['fc_dim'] // 2, config['model']['out_dim']),
        )

    def forward(self, data):
        x = data.x                                  # [N,2] normalized coords
        edge_index = data.edge_index                # undirected (20 edges)
        edge_attr = data.edge_attr                  # [E,1] scaled area (duplicated)
        pos_raw   = getattr(data, 'pos_raw', data.pos)  # real coords for L, cos

        h = self.enc(x)
        for blk in self.blocks:
            h = blk(h, edge_index, edge_attr, pos_raw)

        y_hat = self.dec(h)                         # [N,2] displacements (scaled if you trained scaled)

        # If you scaled targets during training, bc must be in the same scaled units.
        if hasattr(data, "fixed"):
            # Optional: if you later use nonzero BCs, pass data.bc here (must be scaled like y)
            y_hat = apply_dirichlet(y_hat, data.fixed, bc=None)

        return y_hat


class AGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enc = MLP(config['model']['in_dim'],
                       config['model']['fc_dim'],
                       hidden=config['model']['fc_dim'],
                       dropout=config['model']['dropout'])
        self.gnn_blocks = nn.ModuleList([MPNNBlock(config['model']['fc_dim'],
                                               config['model']['edge_dim'],
                                               aggr='mean',
                                               dropout=config['model']['dropout'])
                                     for _ in range(config['model']['N_layer'])])

        self.global_blocks = nn.ModuleList([
            GlobalAttentionBlock(width=config['model']['fc_dim'], hidden=config['model']['fc_dim'], norm=True)
            for _ in range(config['model']['N_layer'])
        ])

        self.dec = nn.Sequential(
            nn.GELU(),
            nn.Linear(config['model']['fc_dim'], config['model']['fc_dim'] // 2),
            nn.GELU(),
            nn.Linear(config['model']['fc_dim'] // 2, config['model']['out_dim']),
        )

    def forward(self, data, return_last_attn=False):
        x = data.x                                  # [N,2] normalized coords
        edge_index = data.edge_index                # undirected (20 edges)
        edge_attr = data.edge_attr                  # [E,1] scaled area (duplicated)
        pos_raw   = getattr(data, 'pos_raw', data.pos)  # real coords for L, cos
        batch     = getattr(data, 'batch', None)

        last_alpha = None

        L = len(self.global_blocks)
        h = self.enc(x)
        for li, (gnn_blk, glob_blk) in enumerate(zip(self.gnn_blocks, self.global_blocks)):
            h = gnn_blk(h, edge_index, edge_attr, pos_raw)

            is_last = (li == L - 1) and return_last_attn
            if is_last:
                h, last_alpha = glob_blk(h, batch=batch, return_attention=True)
            else:
                h = glob_blk(h, batch=batch)

        y_hat = self.dec(h)                         # [N,2] displacements (scaled if you trained scaled)

        # If you scaled targets during training, bc must be in the same scaled units.
        if hasattr(data, "fixed"):
            # Optional: if you later use nonzero BCs, pass data.bc here (must be scaled like y)
            y_hat = apply_dirichlet(y_hat, data.fixed, bc=None)

        if return_last_attn:
            return y_hat, {"alpha": last_alpha, "batch": batch}
        return y_hat


class GNO(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(config['model']['in_dim'], config['model']['fc_dim']), nn.GELU())
        self.gno_blocks = nn.ModuleList([
            GNOBlock(config['model']['fc_dim'],
                     config['model']['edge_dim'],
                     config['model']['fc_dim'],
                     aggr='sum', norm=True,
                     symmetrize=True)
            for _ in range(config['model']['N_layer'])
        ])
        self.dec = nn.Sequential(nn.GELU(), nn.Linear(config['model']['fc_dim'], config['model']['out_dim']))

    def forward(self, data):
        x = self.enc(data.x)  # [N, width]
        for blk in self.gno_blocks:
            x = blk(x, data.edge_index, data.edge_attr)
        y = self.dec(x)
        # If you scaled targets during training, bc must be in the same scaled units.
        if hasattr(data, "fixed"):
            # Optional: if you later use nonzero BCs, pass data.bc here (must be scaled like y)
            y = apply_dirichlet(y, data.fixed, bc=None)

        return y


class GlobalAttentionBlock(nn.Module):
    """
    Global communication: each graph gets a context vector by attending
    over its nodes, then this context is broadcast back and used to update
    node embeddings (residual + LayerNorm).
    """
    def __init__(self, width, hidden=None, norm=True):
        super().__init__()
        self.width = width
        hidden = hidden or width

        # Attention scoring: x_i -> scalar score
        self.att_mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

        # How to mix (x_i, context_g) into updated x_i
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * width, hidden),
            nn.GELU(),
            nn.Linear(hidden, width)
        )

        self.ln = nn.LayerNorm(width) if norm else nn.Identity()

    def forward(self, x, batch=None, return_attention=False):
        """
        x: [N, C]
        batch: [N] graph index per node (0..B-1); if None, assume single graph
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # scores: [N, 1] -> [N]
        scores = self.att_mlp(x).squeeze(-1)
        # softmax per graph
        alpha = softmax(scores, batch)  # [N]

        # weighted sum per graph -> context: [B, C]
        x_weighted = x * alpha.unsqueeze(-1)  # [N, C]
        context = scatter(x_weighted, batch, dim=0, reduce='sum')  # [B, C]

        # broadcast context back to nodes
        ctx_per_node = context[batch]  # [N, C]

        # update node features with residual
        msg = self.msg_mlp(torch.cat([x, ctx_per_node], dim=-1))  # [N, C]
        out = self.ln(x + msg)

        if return_attention:
            return out, alpha
        return out


class AGNO(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_dim  = config['model']['in_dim']
        fc_dim  = config['model']['fc_dim']
        out_dim = config['model']['out_dim']
        edge_dim = config['model']['edge_dim']
        n_layer = config['model']['N_layer']

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(in_dim, fc_dim),
            nn.GELU()
        )

        # Local GNO + Global blocks
        self.gno_blocks = nn.ModuleList([
            GNOBlock(
                width=fc_dim,
                ker_in=edge_dim,
                ker_width=fc_dim,
                aggr='sum',
                norm=True,
                symmetrize=True
            )
            for _ in range(n_layer)
        ])

        self.global_blocks = nn.ModuleList([
            GlobalAttentionBlock(width=fc_dim, hidden=fc_dim, norm=True)
            for _ in range(n_layer)
        ])

        # Decoder
        self.dec = nn.Sequential(
            nn.GELU(),
            nn.Linear(fc_dim, out_dim)
        )

    def forward(self, data, return_last_attn=False):
        # OPTIONAL: if you want to inject fixed BC as a feature:
        # if hasattr(data, "fixed"):
        #     fixed_mask = data.fixed.float()  # [N, dof] or [N,1]; adjust in_dim accordingly
        #     x_in = torch.cat([data.x, fixed_mask], dim=-1)
        # else:
        #     x_in = data.x
        # x = self.enc(x_in)

        x = self.enc(data.x)         # [N, fc_dim]

        edge_index = data.edge_index
        edge_attr  = data.edge_attr
        batch      = getattr(data, 'batch', None)

        last_alpha = None

        L = len(self.global_blocks)
        for li, (gno_blk, glob_blk) in enumerate(zip(self.gno_blocks, self.global_blocks)):
            # Local message passing (1-hop)
            x = gno_blk(x, edge_index, edge_attr)

            # Global communication across the entire graph
            is_last = (li == L - 1) and return_last_attn
            if is_last:
                x, last_alpha = glob_blk(x, batch=batch, return_attention=True)
            else:
                x = glob_blk(x, batch=batch)

        y = self.dec(x)

        # Apply Dirichlet BCs on the output (still needed!)
        if hasattr(data, "fixed"):
            # bc=None here means zero-displacement BCs; adjust if you have non-zero
            y = apply_dirichlet(y, data.fixed, bc=None)

        if return_last_attn:
            return y, {"alpha": last_alpha, "batch": batch}
        return y


class New_model(nn.Module):

    def __init__(self, config):
        super().__init__()

    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)
        z_coor: (B, M)
        shape_coor: (B, M'', 2)

        return u, v: (B, M)
        '''
        u = None
        v = None

        return u, v
