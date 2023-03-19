from __future__ import print_function
import torch.nn as nn
import torch
import pprint
import torch.distributions as D
from torch.nn.modules.activation import ReLU
from torch.nn.modules.container import Sequential

STD = 0.5


class HM(nn.Module):
    def __init__(self, arch, **kwargs):
        super().__init__()
        self.config = kwargs['kwargs']
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.config)
        self.batch_size = self.config['batch_size']

        if arch == 1:
            dim = 2
        elif arch == 2:
            dim = 8
        z = 2
        self.infnet = nn.ModuleList([
            nn.Sequential(nn.Linear(2, dim), nn.ReLU(), nn.Linear(dim, z)),
            nn.Sequential(nn.Linear(z + 1, dim), nn.ReLU(), nn.Linear(dim, z)),
            nn.Sequential(nn.Linear(1, dim), nn.ReLU(), nn.Linear(dim, z)),
            nn.Sequential(nn.Linear(z, dim), nn.ReLU(), nn.Linear(dim, z))
        ])
        self.gennet = nn.ModuleList([
            nn.Linear(1, z, bias=False),
            nn.Sequential(nn.Linear(z, dim), nn.ReLU(), nn.Linear(dim, z)),
            nn.Sequential(nn.Linear(z, dim), nn.ReLU(), nn.Linear(dim, 1)),
            nn.Sequential(nn.Linear(1, dim), nn.ReLU(), nn.Linear(dim, z)),
            nn.Sequential(nn.Linear(z, dim), nn.ReLU(), nn.Linear(dim, z)),
            nn.Sequential(nn.Linear(z, dim), nn.ReLU(), nn.Linear(dim, 1)),
        ])

        self.cuda()

    def sample_binary(self, p, noise=None):
        if noise is None:
            noise = torch.rand_like(p, dtype=torch.float,
                                    requires_grad=False).cuda()
        assert noise.shape == p.shape
        z = (p > noise).type(torch.float)

        return z, noise

    def sample_gaussian(self, mu):

        noise = torch.zeros_like(mu, dtype=torch.float,
                                 requires_grad=False).normal_().cuda()

        return mu + STD * noise

    def generate(self):
        sm = nn.Sigmoid()

        tmp = torch.ones((self.batch_size, 1),
                         dtype=torch.float,
                         requires_grad=True).cuda()

        tmp = self.gennet[0](tmp)
        pz4 = sm(tmp)
        z4, nz4 = self.sample_binary(pz4)

        tmp = self.gennet[1](z4)
        pz3 = sm(tmp)
        z3, nz3 = self.sample_binary(pz3)

        muy2 = self.gennet[2](z3)
        y2 = self.sample_gaussian(muy2)

        tmp = self.gennet[3](y2)
        pz2 = sm(tmp)
        z2, nz2 = self.sample_binary(pz2)

        tmp = self.gennet[4](z2)
        pz1 = sm(tmp)
        z1, nz1 = self.sample_binary(pz1)

        muy1 = self.gennet[5](z1)
        y1 = self.sample_gaussian(muy1)

        return {
            'z': [z1, z2, z3, z4],
            'y': [y1, y2],
            'nz': [nz1, nz2, nz3, nz4],
            'pz': [pz1, pz2, pz3, pz4],
            'muy': [muy1, muy2]
        }

    def generate_wake(self, param):
        z1, z2, z3, z4 = param['z']
        y1, y2 = param['y']

        tmp = torch.ones((self.batch_size, 1),
                         dtype=torch.float,
                         requires_grad=True).cuda()

        l4 = self.gennet[0](tmp)

        l3 = self.gennet[1](z4)

        muy2 = self.gennet[2](z3)

        l2 = self.gennet[3](y2)

        l1 = self.gennet[4](z2)

        muy1 = self.gennet[5](z1)

        return {'logitz': [l1, l2, l3, l4], 'muy': [muy1, muy2]}

    def generate_distill(self, param):
        nz1, nz2, nz3, nz4 = param['nz']
        y1, y2 = param['y']

        sm = nn.Sigmoid()

        tmp = torch.ones((self.batch_size, 1),
                         dtype=torch.float,
                         requires_grad=True).cuda()

        l4 = self.gennet[0](tmp)
        pz4 = sm(l4)
        z4, nz4 = self.sample_binary(pz4, nz4)

        l3 = self.gennet[1](z4)
        pz3 = sm(l3)
        z3, nz3 = self.sample_binary(pz3, nz3)

        muy2 = self.gennet[2](z3)

        l2 = self.gennet[3](y2)
        pz2 = sm(l2)
        z2, nz2 = self.sample_binary(pz2, nz2)

        l1 = self.gennet[4](z2)
        pz1 = sm(l1)
        z1, nz1 = self.sample_binary(pz1, nz1)

        muy1 = self.gennet[5](z1)

        return {'logitz': [l1, l2, l3, l4], 'muy': [muy1, muy2]}

    def infer(self, param):
        y1, y2 = param['y']

        sm = nn.Sigmoid()

        tmp = torch.cat([y1, y2], dim=1)
        tmp = self.infnet[0](tmp)
        pz1 = sm(tmp)
        z1, _ = self.sample_binary(pz1)

        tmp = torch.cat([y2, z1], dim=1)
        tmp = self.infnet[1](tmp)
        pz2 = sm(tmp)
        z2, _ = self.sample_binary(pz2)

        tmp = self.infnet[2](y2)
        pz3 = sm(tmp)
        z3, _ = self.sample_binary(pz3)

        tmp = self.infnet[3](z3)
        pz4 = sm(tmp)
        z4, _ = self.sample_binary(pz4)

        param['z'] = [z1, z2, z3, z4]

        return param

    def infer_sleep(self, param):
        z1, z2, z3, z4 = param['z']
        y1, y2 = param['y']

        tmp = torch.cat([y1, y2], dim=1)
        l1 = self.infnet[0](tmp)

        tmp = torch.cat([y2, z1], dim=1)
        l2 = self.infnet[1](tmp)

        l3 = self.infnet[2](y2)

        l4 = self.infnet[3](z3)

        return {'logitz': [l1, l2, l3, l4]}

    def wake(self, param):
        param_inf = self.infer(param)

        param_gen = self.generate_wake(param_inf)

        l1, l2, l3, l4 = param_gen['logitz']
        muy1, muy2 = param_gen['muy']
        z1, z2, z3, z4 = param_inf['z']
        y1, y2 = param_inf['y']

        bcel = nn.BCEWithLogitsLoss(reduction='sum')
        loss_z = bcel(l1, z1) + bcel(l2, z2) + bcel(l3, z3) + bcel(l4, z4)
        loss_z = loss_z / self.batch_size

        loss_y = -D.Normal(muy1, STD).log_prob(y1).sum() - D.Normal(
            muy2, STD).log_prob(y2).sum()
        loss_y = loss_y / self.batch_size

        loss = loss_z + loss_y

        return loss, loss_z, loss_y

    def sleep(self):
        param_gen = self.generate()

        param_inf = self.infer_sleep(param_gen)

        l1, l2, l3, l4 = param_inf['logitz']
        z1, z2, z3, z4 = param_gen['z']

        bcel = nn.BCEWithLogitsLoss(reduction='sum')
        loss = bcel(l1, z1) + bcel(l2, z2) + bcel(l3, z3) + bcel(l4, z4)
        loss = loss / self.batch_size

        return loss
