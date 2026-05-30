import torch
import torch.nn as nn

class DG(nn.Module):

    def __init__(self, config):
        super().__init__()

        # branch network
        trunk_layers = [nn.Linear(2, config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(config['model']['fc_dim'], config['model']['fc_dim']))
        self.branch = nn.Sequential(*trunk_layers)

    def forward(self, shape_coor, shape_flag):
        '''
        shape_coor: (B, M'', 2)
        shape_flag: (B, M'')

        return u: (B, 1, F)
        '''

        # get the first kernel
        enc = self.branch(shape_coor)  # (B, M, F)
        enc_masked = enc * shape_flag.unsqueeze(-1)  # (B, M, F)
        Domain_enc = torch.sum(enc_masked, 1, keepdim=True) / torch.sum(shape_flag.unsqueeze(-1), 1,
                                                                        keepdim=True)  # (B, 1, F)

        return Domain_enc


class GANO(nn.Module):

    def __init__(self, config):
        super().__init__()

        # define the geometry encoder
        self.DG = DG(config)

        # branch network
        trunk_layers = [nn.Linear(3, 2 * config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim']))
        self.branch = nn.Sequential(*trunk_layers)

        # parlifting layer
        self.xy_lift1 = nn.Linear(2, config['model']['fc_dim'])
        self.xy_lift2 = nn.Linear(2, config['model']['fc_dim'])
        self.xy_lift3 = nn.Linear(2, config['model']['fc_dim'])

        # trunk network 1
        self.FC1u = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC2u = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC3u = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC4u = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC5u = nn.Linear(2 * config['model']['fc_dim'], 1)
        self.act = nn.Tanh()

        # trunk network 2
        self.FC1v = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC2v = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC3v = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC4v = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC5v = nn.Linear(2 * config['model']['fc_dim'], 1)

        # trunk network 3
        self.FC1theta = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC2theta = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC3theta = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC4theta = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC5theta = nn.Linear(2 * config['model']['fc_dim'], 1)

    def predict_geometry_embedding(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        Domain_enc = self.DG(shape_coor, shape_flag)  # (B,1,F)

        return Domain_enc

    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)
        z_coor: (B, M)
        shape_coor: (B, M'', 2)

        return u: (B, M)
        '''

        # extract number of points
        B, mD = x_coor.shape

        # forward to get the domain embedding
        Domain_enc = self.DG(shape_coor, shape_flag)  # (B,1,F)

        # concat coors
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # lift the dimension of coordinate embedding
        xy_local_u = self.xy_lift1(xy)  # (B,M,F)
        xy_local_v = self.xy_lift2(xy)  # (B,M,F)
        xy_local_theta = self.xy_lift3(xy)  # (B,M,F)

        # combine with global embedding
        xy_global_u = torch.cat((xy_local_u, Domain_enc.repeat(1, mD, 1)), -1)  # (B,M,2F)
        xy_global_v = torch.cat((xy_local_v, Domain_enc.repeat(1, mD, 1)), -1)  # (B,M,2F)
        xy_global_theta = torch.cat((xy_local_theta, Domain_enc.repeat(1, mD, 1)), -1)  # (B,M,2F)

        # get the kernels
        enc = self.branch(par)  # (B, M, F)
        enc_masked = enc * par_flag.unsqueeze(-1)  # (B, M, F)
        enc = torch.amax(enc_masked, 1, keepdim=True)  # (B, 1, F)

        # predict u
        u = self.FC1u(xy_global_u)  # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC2u(u)  # (B,M,F)
        u = self.act(u)
        u = u * enc
        u = self.FC3u(u)  # (B,M,F)
        u = self.act(u)
        # u = u * enc
        u = self.FC4u(u)  # (B,M,F)
        # u = self.act(u)
        u = torch.mean(u * enc, -1)  # (B, M)

        # predict v
        v = self.FC1v(xy_global_v)  # (B,M,F)
        v = self.act(v)
        v = v * enc
        v = self.FC2v(v)  # (B,M,F)
        v = self.act(v)
        v = v * enc
        v = self.FC3v(v)  # (B,M,F)
        v = self.act(v)
        # v = v * enc
        v = self.FC4v(v)  # (B,M,F)
        # v = self.act(v)
        v = torch.mean(v * enc, -1)  # (B, M)

        # predict v
        theta = self.FC1theta(xy_global_theta)  # (B,M,F)
        theta = self.act(theta)
        theta = theta * enc
        theta = self.FC2theta(theta)  # (B,M,F)
        theta = self.act(theta)
        theta = theta * enc
        theta = self.FC3theta(theta)  # (B,M,F)
        theta = self.act(theta)
        # v = v * enc
        theta = self.FC4theta(theta)  # (B,M,F)
        # v = self.act(v)
        theta = torch.mean(theta * enc, -1)  # (B, M)

        return u, v, theta


class New_model_frame(nn.Module):

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

