import torch
import torch.nn as nn


class DG(nn.Module):

    def __init__(self, config):
        super().__init__()

        # branch network
        trunk_layers = [nn.Linear(3, config['model']['fc_dim']), nn.Tanh()]
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
        trunk_layers = [nn.Linear(4, 2 * config['model']['fc_dim']), nn.Tanh()]
        for _ in range(config['model']['N_layer'] - 1):
            trunk_layers.append(nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim']))
            trunk_layers.append(nn.Tanh())
        trunk_layers.append(nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim']))
        self.branch = nn.Sequential(*trunk_layers)

        # parlifting layer
        self.xyz_lift1 = nn.Linear(3, config['model']['fc_dim'])
        self.xyz_lift2 = nn.Linear(3, config['model']['fc_dim'])
        self.xyz_lift3 = nn.Linear(3, config['model']['fc_dim'])
        self.xyz_lift4 = nn.Linear(3, config['model']['fc_dim'])
        self.xyz_lift5 = nn.Linear(3, config['model']['fc_dim'])
        self.xyz_lift6 = nn.Linear(3, config['model']['fc_dim'])

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
        self.FC1w = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC2w = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC3w = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC4w = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC5w = nn.Linear(2 * config['model']['fc_dim'], 1)

        # trunk network 4
        self.FC1phix = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC2phix = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC3phix = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC4phix = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC5phix = nn.Linear(2 * config['model']['fc_dim'], 1)

        # trunk network 5
        self.FC1phiy = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC2phiy = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC3phiy = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC4phiy = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC5phiy = nn.Linear(2 * config['model']['fc_dim'], 1)

        # trunk network 6
        self.FC1phiz = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC2phiz = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC3phiz = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC4phiz = nn.Linear(2 * config['model']['fc_dim'], 2 * config['model']['fc_dim'])
        self.FC5phiz = nn.Linear(2 * config['model']['fc_dim'], 1)

    def predict_geometry_embedding(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        Domain_enc = self.DG(shape_coor, shape_flag)  # (B,1,F)

        return Domain_enc

    def forward(self, x_coor, y_coor, z_coor, par, par_flag, shape_coor, shape_flag):
        '''
        par: (B, M', 4)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)
        z_coor: (B, M)
        shape_coor: (B, M'', 3)

        return u: (B, M)
        '''

        # extract number of points
        B, mD = x_coor.shape

        # forward to get the domain embedding
        Domain_enc = self.DG(shape_coor, shape_flag)  # (B,1,F)

        # concat coors
        xyz = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1), z_coor.unsqueeze(-1)), -1)

        # lift the dimension of coordinate embedding
        xyz_local_u = self.xyz_lift1(xyz)  # (B,M,F)
        xyz_local_v = self.xyz_lift2(xyz)  # (B,M,F)
        xyz_local_w = self.xyz_lift3(xyz)  # (B,M,F)
        xyz_local_phix = self.xyz_lift4(xyz)  # (B,M,F)
        xyz_local_phiy = self.xyz_lift5(xyz)  # (B,M,F)
        xyz_local_phiz = self.xyz_lift6(xyz)  # (B,M,F)

        # combine with global embedding
        xyz_global_u = torch.cat((xyz_local_u, Domain_enc.repeat(1, mD, 1)), -1)  # (B,M,2F)
        xyz_global_v = torch.cat((xyz_local_v, Domain_enc.repeat(1, mD, 1)), -1)  # (B,M,2F)
        xyz_global_w = torch.cat((xyz_local_w, Domain_enc.repeat(1, mD, 1)), -1)  # (B,M,2F)
        xyz_global_phix = torch.cat((xyz_local_phix, Domain_enc.repeat(1, mD, 1)), -1)  # (B,M,2F)
        xyz_global_phiy = torch.cat((xyz_local_phiy, Domain_enc.repeat(1, mD, 1)), -1)  # (B,M,2F)
        xyz_global_phiz = torch.cat((xyz_local_phiz, Domain_enc.repeat(1, mD, 1)), -1)  # (B,M,2F)

        # get the kernels
        enc = self.branch(par)  # (B, M, F)
        enc_masked = enc * par_flag.unsqueeze(-1)  # (B, M, F)
        enc = torch.amax(enc_masked, 1, keepdim=True)  # (B, 1, F)

        # predict u
        u = self.FC1u(xyz_global_u)  # (B,M,F)
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
        v = self.FC1v(xyz_global_v)  # (B,M,F)
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

        # predict w
        w = self.FC1w(xyz_global_w)  # (B,M,F)
        w = self.act(w)
        w = w * enc
        w = self.FC2w(w)  # (B,M,F)
        w = self.act(w)
        w = w * enc
        w = self.FC3w(w)  # (B,M,F)
        w = self.act(w)
        # v = v * enc
        w = self.FC4w(w)  # (B,M,F)
        # v = self.act(v)
        w = torch.mean(w * enc, -1)  # (B, M)

        # predict phix
        phix = self.FC1phix(xyz_global_phix)  # (B,M,F)
        phix = self.act(phix)
        phix = phix * enc
        phix = self.FC2phix(phix)  # (B,M,F)
        phix = self.act(phix)
        phix = phix * enc
        phix = self.FC3phix(phix)  # (B,M,F)
        phix = self.act(phix)
        # v = v * enc
        phix = self.FC4phix(phix)  # (B,M,F)
        # v = self.act(v)
        phix = torch.mean(phix * enc, -1)  # (B, M)

        # predict phiy
        phiy = self.FC1phiy(xyz_global_phiy)  # (B,M,F)
        phiy = self.act(phiy)
        phiy = phiy * enc
        phiy = self.FC2phiy(phiy)  # (B,M,F)
        phiy = self.act(phiy)
        phiy = phiy * enc
        phiy = self.FC3phiy(phiy)  # (B,M,F)
        phiy = self.act(phiy)
        # v = v * enc
        phiy = self.FC4phiy(phiy)  # (B,M,F)
        # v = self.act(v)
        phiy = torch.mean(phiy * enc, -1)  # (B, M)

        # predict phiz
        phiz = self.FC1phiz(xyz_global_phiz)  # (B,M,F)
        phiz = self.act(phiz)
        phiz = phiz * enc
        phiz = self.FC2phiz(phiz)  # (B,M,F)
        phiz = self.act(phiz)
        phiz = phiz * enc
        phiz = self.FC3phiz(phiz)  # (B,M,F)
        phiz = self.act(phiz)
        # v = v * enc
        phiz = self.FC4phiz(phiz)  # (B,M,F)
        # v = self.act(v)
        phiz = torch.mean(phiz * enc, -1)  # (B, M)

        return u, v, w, phix, phiy, phiz


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
