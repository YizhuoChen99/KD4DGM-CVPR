from torch import nn
import torch
from collections import OrderedDict
import pprint
import torch.distributions as D


class VRNN(nn.Module):
    def __init__(self, arch, **kwargs):

        super().__init__()

        self.config = kwargs['kwargs']
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.config)

        ### General parameters
        self.batch_size = self.config['batch_size']
        self.gen_seq_len = 628
        self.x_dim = 3
        self.z_dim = 16
        self.y_dim = self.x_dim
        activation = 'relu'
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')
        self.device = 0

        ### loss
        self.beta = 1
        self.STD = self.config['STD']

        if arch == 1:

            ### Feature extractors
            self.dense_x = [128, 128]
            self.dense_z = [128, 128]
            ### Dense layers
            self.dense_hx_z = [128, 128]
            self.dense_hz_x = [128, 128]
            self.dense_h_z = [128, 128]
            ### RNN
            self.dim_RNN = 600
            self.num_RNN = 1

        elif arch == 2:

            ### Feature extractors
            self.dense_x = [512, 512]
            self.dense_z = [512, 512]
            ### Dense layers
            self.dense_hx_z = [512, 512]
            self.dense_hz_x = [512, 512]
            self.dense_h_z = [512, 512]
            ### RNN
            self.dim_RNN = 1200
            self.num_RNN = 1

        self.build()
        self.to(self.device)

    def build(self):

        # x
        dic_layers = OrderedDict()
        if len(self.dense_x) == 0:
            dim_feature_x = self.x_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_feature_x = self.dense_x[-1]
            for n in range(len(self.dense_x)):
                if n == 0:
                    dic_layers['linear' + str(n)] = nn.Linear(
                        self.x_dim, self.dense_x[n])
                else:
                    dic_layers['linear' + str(n)] = nn.Linear(
                        self.dense_x[n - 1], self.dense_x[n])
                dic_layers['activation' + str(n)] = self.activation
        self.feature_extractor_x = nn.Sequential(dic_layers)
        # z
        dic_layers = OrderedDict()
        if len(self.dense_z) == 0:
            dim_feature_z = self.z_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_feature_z = self.dense_z[-1]
            for n in range(len(self.dense_z)):
                if n == 0:
                    dic_layers['linear' + str(n)] = nn.Linear(
                        self.z_dim, self.dense_z[n])
                else:
                    dic_layers['linear' + str(n)] = nn.Linear(
                        self.dense_z[n - 1], self.dense_z[n])
                dic_layers['activation' + str(n)] = self.activation
        self.feature_extractor_z = nn.Sequential(dic_layers)

        # 1. h_t, x_t to z_t (Inference)
        dic_layers = OrderedDict()
        if len(self.dense_hx_z) == 0:
            dim_hx_z = self.dim_RNN + dim_feature_x
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_hx_z = self.dense_hx_z[-1]
            for n in range(len(self.dense_hx_z)):
                if n == 0:
                    dic_layers['linear' + str(n)] = nn.Linear(
                        self.dense_x[-1] + self.dim_RNN, self.dense_hx_z[n])
                else:
                    dic_layers['linear' + str(n)] = nn.Linear(
                        self.dense_hx_z[n - 1], self.dense_hx_z[n])
                dic_layers['activation' + str(n)] = self.activation
        self.mlp_hx_z = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(dim_hx_z, self.z_dim)
        self.inf_logvar = nn.Linear(dim_hx_z, self.z_dim)

        # 2. h_t to z_t (Generation z)
        dic_layers = OrderedDict()
        if len(self.dense_h_z) == 0:
            dim_h_z = self.dim_RNN
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_h_z = self.dense_h_z[-1]
            for n in range(len(self.dense_h_z)):
                if n == 0:
                    dic_layers['linear' + str(n)] = nn.Linear(
                        self.dim_RNN, self.dense_h_z[n])
                else:
                    dic_layers['linear' + str(n)] = nn.Linear(
                        self.dense_h_z[n - 1], self.dense_h_z[n])
                dic_layers['activation' + str(n)] = self.activation
        self.mlp_h_z = nn.Sequential(dic_layers)
        self.prior_mean = nn.Linear(dim_h_z, self.z_dim)
        self.prior_logvar = nn.Linear(dim_h_z, self.z_dim)

        # 3. h_t, z_t to x_t (Generation x)
        dic_layers = OrderedDict()
        if len(self.dense_hz_x) == 0:
            dim_hz_x = self.dim_RNN + dim_feature_z
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_hz_x = self.dense_hz_x[-1]
            for n in range(len(self.dense_hz_x)):
                if n == 0:
                    dic_layers['linear' + str(n)] = nn.Linear(
                        self.dim_RNN + dim_feature_z, self.dense_hz_x[n])
                else:
                    dic_layers['linear' + str(n)] = nn.Linear(
                        self.dense_hz_x[n - 1], self.dense_hz_x[n])
                dic_layers['activation' + str(n)] = self.activation
        self.mlp_hz_x = nn.Sequential(dic_layers)
        self.gen_linear = nn.Linear(dim_hz_x, self.y_dim)

        self.rnn = nn.LSTM(dim_feature_x + dim_feature_z, self.dim_RNN,
                           self.num_RNN)

    def reparameterization(self, mean, logvar, noise=None, scale=1):

        std = torch.exp(0.5 * logvar)
        if noise is None:
            noise = torch.randn_like(std)
        noise = noise * scale

        return torch.addcmul(mean, noise, std), noise

    def reparameterization_x(self, y, scale=1, scale_binary=1):
        binary_x = nn.Sigmoid()(y[:, :, 0:1])
        if scale_binary != 1:
            binary_x = 1 / (1 +
                            ((1 - binary_x) / binary_x)**(1 / scale_binary**2))
        noise = torch.rand_like(binary_x)
        binary_x = (binary_x > noise).type(torch.float)

        gauss_x = y[:, :, 1:3]
        noise = torch.randn_like(gauss_x)
        noise = noise * scale
        gauss_x = torch.addcmul(gauss_x, noise,
                                torch.full_like(gauss_x, self.STD))

        res = torch.cat([binary_x, gauss_x], dim=2)
        return res

    def generation_x(self, feature_zt, h_t):

        dec_input = torch.cat((feature_zt, h_t), 2)
        dec_output = self.mlp_hz_x(dec_input)
        y_t = self.gen_linear(dec_output)

        return y_t

    def generation_z(self, h):

        prior_output = self.mlp_h_z(h)
        mean_prior = self.prior_mean(prior_output)
        logvar_prior = self.prior_logvar(prior_output)

        return mean_prior, logvar_prior

    def inference(self, feature_xt, h_t):

        enc_input = torch.cat((feature_xt, h_t), 2)
        enc_output = self.mlp_hx_z(enc_input)
        mean_zt = self.inf_mean(enc_output)
        logvar_zt = self.inf_logvar(enc_output)

        return mean_zt, logvar_zt

    def recurrence(self, feature_xt, feature_zt, h_t, c_t):

        rnn_input = torch.cat((feature_xt, feature_zt), -1)
        _, (h_tp1, c_tp1) = self.rnn(rnn_input, (h_t, c_t))

        return h_tp1, c_tp1

    def loss_reduction(self, loss_pad, lengths):
        losses = []
        for b in range(len(lengths)):
            losses.append(torch.sum(loss_pad[:lengths[b], b]))
        loss = sum(losses) / sum(lengths)
        return loss

    def kl_gaussian(self, p_mu, p_logvar, q_mu, q_logvar):

        res = -0.5 * (1 + p_logvar - q_logvar -
                      torch.div(p_logvar.exp() +
                                (p_mu - q_mu).pow(2), q_logvar.exp()))

        return res

    def kl_gaussian_std(self, p_mu, p_std, q_mu, q_std):

        p = D.Normal(p_mu, p_std)
        q = D.Normal(q_mu, q_std)

        return D.kl_divergence(p, q)

    def kl_bernoulli(self, q_logits, p):

        res = nn.BCEWithLogitsLoss(reduction='none')(q_logits, p)

        return res

    def forward(self, x, lengths):
        # x: n_points_line * batch_size * 3
        # lengths: list of int: batch_size
        batch_size = self.batch_size
        seq_len = x.shape[0]

        # create variable holders
        z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_logvar = torch.zeros(
            (seq_len, batch_size, self.z_dim)).to(self.device)
        y = torch.zeros((seq_len, batch_size, self.y_dim)).to(self.device)
        z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        h = torch.zeros((seq_len, batch_size, self.dim_RNN)).to(self.device)

        # init rnn h_0 z_0
        h_t = torch.zeros(self.num_RNN, batch_size,
                          self.dim_RNN).to(self.device)
        c_t = torch.zeros(self.num_RNN, batch_size,
                          self.dim_RNN).to(self.device)

        # inference and reconsturction
        feature_x = self.feature_extractor_x(x)
        for t in range(seq_len):
            feature_xt = feature_x[t, :, :].unsqueeze(0)
            # 1 * B * dim_fx
            h_t_last = h_t.view(self.num_RNN, 1, batch_size,
                                self.dim_RNN)[-1, :, :, :]
            # 1 * B * dim_h

            mean_zt, logvar_zt = self.inference(feature_xt, h_t_last)
            z_t, _ = self.reparameterization(mean_zt, logvar_zt)
            # 1 * B * dim_z

            feature_zt = self.feature_extractor_z(z_t)
            # 1 * B * dim_fz
            y_t = self.generation_x(feature_zt, h_t_last)
            # 1 * B * dim_x

            h_t, c_t = self.recurrence(feature_xt, feature_zt, h_t, c_t)
            # num_rnn * B * dim_h

            # record
            z_mean[t, :, :] = torch.squeeze(mean_zt)
            z_logvar[t, :, :] = torch.squeeze(logvar_zt)
            z[t, :, :] = torch.squeeze(z_t)
            y[t, :, :] = torch.squeeze(y_t)
            h[t, :, :] = torch.squeeze(h_t_last)

        # generation
        z_mean_p, z_logvar_p = self.generation_z(h)
        # n_points_line * B * dim_z

        # Calculate unreduced loss
        loss_recon_binary = self.kl_bernoulli(y[:, :, 0], x[:, :, 0])
        loss_recon_binary = loss_recon_binary.squeeze()
        loss_recon_gauss = -D.Normal(y[:, :, 1:3], self.STD).log_prob(x[:, :,
                                                                        1:3])
        loss_recon_gauss = torch.sum(loss_recon_gauss, dim=2)
        loss_KLD = self.kl_gaussian(z_mean, z_logvar, z_mean_p, z_logvar_p)
        loss_KLD = torch.sum(loss_KLD, dim=2)
        # n_points_line * B

        # loss reduction
        loss_recon_binary = self.loss_reduction(loss_recon_binary, lengths)
        loss_recon_gauss = self.loss_reduction(loss_recon_gauss, lengths)
        loss_KLD = self.loss_reduction(loss_KLD, lengths)

        loss = loss_recon_binary + loss_recon_gauss + self.beta * loss_KLD

        return loss, {
            'loss_recon_binary_mean': loss_recon_binary,
            'loss_recon_gauss_mean': loss_recon_gauss,
            'loss_KLD_mean': loss_KLD
        }

    def generate(self, scale_z=1, scale_x=1, scale_x_binary=1):
        batch_size = self.batch_size
        seq_len = self.gen_seq_len

        # create variable holders
        z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_logvar = torch.zeros(
            (seq_len, batch_size, self.z_dim)).to(self.device)
        x = torch.zeros((seq_len, batch_size, self.x_dim)).to(self.device)
        y = torch.zeros((seq_len, batch_size, self.y_dim)).to(self.device)
        z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        nz = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)

        # init rnn h_0 z_0
        h_t = torch.zeros(self.num_RNN, batch_size,
                          self.dim_RNN).to(self.device)
        c_t = torch.zeros(self.num_RNN, batch_size,
                          self.dim_RNN).to(self.device)

        # inference and reconsturction
        for t in range(seq_len):
            h_t_last = h_t.view(self.num_RNN, 1, batch_size,
                                self.dim_RNN)[-1, :, :, :]
            # 1 * B * dim_h

            mean_zt, logvar_zt = self.generation_z(h_t_last)
            z_t, nz_t = self.reparameterization(mean_zt,
                                                logvar_zt,
                                                scale=scale_z)
            # 1 * B * dim_z

            feature_zt = self.feature_extractor_z(z_t)
            # 1 * B * dim_fz
            y_t = self.generation_x(feature_zt, h_t_last)
            # 1 * B * dim_x

            x_t = self.reparameterization_x(y_t, scale_x, scale_x_binary)
            # 1 * B * dim_x

            feature_xt = self.feature_extractor_x(x_t)
            # 1 * B * dim_fx

            h_t, c_t = self.recurrence(feature_xt, feature_zt, h_t, c_t)
            # num_rnn * B * dim_h

            # record
            z_mean[t, :, :] = torch.squeeze(mean_zt)
            z_logvar[t, :, :] = torch.squeeze(logvar_zt)
            z[t, :, :] = torch.squeeze(z_t)
            x[t, :, :] = torch.squeeze(x_t)
            y[t, :, :] = torch.squeeze(y_t)
            nz[t, :, :] = torch.squeeze(nz_t)

        return {
            'z_mean': z_mean,
            'z_logvar': z_logvar,
            'x': x,
            'y': y,
            'z': z,
            'nz': nz
        }

    def generate_distill(self, param):
        batch_size = self.batch_size
        seq_len = self.gen_seq_len

        teacher_nz = param['nz']
        teacher_z_mean = param['z_mean']
        teacher_z_logvar = param['z_logvar']
        teacher_x = param['x']
        teacher_y = param['y']

        # create variable holders
        z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_logvar = torch.zeros(
            (seq_len, batch_size, self.z_dim)).to(self.device)
        y = torch.zeros((seq_len, batch_size, self.y_dim)).to(self.device)

        # init rnn h_0 z_0
        h_t = torch.zeros(self.num_RNN, batch_size,
                          self.dim_RNN).to(self.device)
        c_t = torch.zeros(self.num_RNN, batch_size,
                          self.dim_RNN).to(self.device)

        # inference and reconsturction
        for t in range(seq_len):
            h_t_last = h_t.view(self.num_RNN, 1, batch_size,
                                self.dim_RNN)[-1, :, :, :]
            # 1 * B * dim_h

            mean_zt, logvar_zt = self.generation_z(h_t_last)
            nz_t = teacher_nz[t:t + 1, :, :]
            z_t, _ = self.reparameterization(mean_zt, logvar_zt, noise=nz_t)
            # 1 * B * dim_z

            feature_zt = self.feature_extractor_z(z_t)
            # 1 * B * dim_fz
            y_t = self.generation_x(feature_zt, h_t_last)
            # 1 * B * dim_x

            x_t = teacher_x[t:t + 1, :, :]

            feature_xt = self.feature_extractor_x(x_t)
            # 1 * B * dim_fx

            h_t, c_t = self.recurrence(feature_xt, feature_zt, h_t, c_t)
            # num_rnn * B * dim_h

            # record
            z_mean[t, :, :] = torch.squeeze(mean_zt)
            z_logvar[t, :, :] = torch.squeeze(logvar_zt)
            y[t, :, :] = torch.squeeze(y_t)

        # loss
        loss_kl_x_binary = self.kl_bernoulli(y[:, :, 0],
                                             nn.Sigmoid()(teacher_y[:, :, 0]))
        loss_kl_x_binary = loss_kl_x_binary.squeeze()
        loss_kl_x_gauss = self.kl_gaussian_std(teacher_y[:, :, 1:3], self.STD,
                                               y[:, :, 1:3], self.STD)
        loss_kl_x_gauss = torch.sum(loss_kl_x_gauss, dim=2)
        loss_kl_z = self.kl_gaussian(teacher_z_mean, teacher_z_logvar, z_mean,
                                     z_logvar)
        loss_kl_z = torch.sum(loss_kl_z, dim=2)

        loss_kl_x_binary = torch.mean(loss_kl_x_binary)
        loss_kl_x_gauss = torch.mean(loss_kl_x_gauss)
        loss_kl_z = torch.mean(loss_kl_z)

        loss = loss_kl_x_binary + loss_kl_x_gauss + self.config[
            'dzlambda'] * loss_kl_z

        return loss, {
            'loss_kl_x_binary_mean': loss_kl_x_binary,
            'loss_kl_x_gauss_mean': loss_kl_x_gauss,
            'loss_kl_z_mean': loss_kl_z
        }
