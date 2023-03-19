import torch
import torch.nn.functional as F
import torch.distributions as D

from .utils import ones_like

GAUSSIAN_VAR_BREAK = True
# all std are set to be exactly 1.0
EPS_STD = 1e-6
DECODER_STD = 0.1


def set_decoder_std(value):
    global DECODER_STD
    DECODER_STD = value


def WS22_gaussian(p_mu, p_std, q_mu, q_std, layer_reduction='sum'):
    batch_size = p_mu.size(0)

    WS22 = F.mse_loss(
        p_mu.view(batch_size, -1), q_mu.view(batch_size, -1),
        reduction='none') + F.mse_loss(p_std.view(batch_size, -1),
                                       q_std.view(batch_size, -1),
                                       reduction='none')

    reduce_func = torch.sum if layer_reduction == 'sum' else torch.mean
    return reduce_func(WS22, dim=-1)


def kl_gaussian_q_p(q_mu, q_std, p_mu, p_std):
    return kl_gaussian(p_mu, p_std, q_mu, q_std)


def kl_gaussian(p_mu, p_std, q_mu, q_std):
    batch_size = p_mu.size(0)

    p = D.Normal(p_mu.view(batch_size, -1), p_std.view(batch_size, -1))
    q = D.Normal(q_mu.view(batch_size, -1), q_std.view(batch_size, -1))

    return torch.sum(D.kl_divergence(p, q), -1)


def kl_bernoulli(p_logits, q_logits):
    batch_size = p_logits.size(0)
    p_logits = p_logits.view(batch_size, -1)
    q_logits = q_logits.view(batch_size, -1)
    p_prob = F.sigmoid(p_logits)

    kl = F.binary_cross_entropy_with_logits(
        q_logits, p_prob,
        reduction='none') - F.binary_cross_entropy_with_logits(
            p_logits, p_prob, reduction='none')

    # p = D.Bernoulli(logits=p_logits.view(batch_size, -1))
    # q = D.Bernoulli(logits=q_logits.view(batch_size, -1))
    # torch.sum(D.kl_divergence(p, q), -1)

    return torch.sum(kl, -1)


def kl_gaussian_q_N_0_1(p_mu, p_std):
    batch_size = p_mu.size(0)
    p_mu = p_mu.view(batch_size, -1)
    p_std = p_std.view(batch_size, -1)
    p = D.Normal(p_mu, p_std)
    q = D.Normal(torch.zeros_like(p_mu), torch.ones_like(p_std))
    return torch.sum(D.kl_divergence(p, q), -1)


def kl_out(recon_s, recon_t, nll_type):
    ''' helper to get the actual KL evaluation '''
    if nll_type == 'bernoulli':
        kl = kl_bernoulli(recon_t, recon_s)
    elif nll_type == 'gaussian':
        num_half_chans = recon_t.size(1) // 2
        recon_s_mu = recon_s[:, 0:num_half_chans, :, :]
        recon_s_logvar = recon_s[:, num_half_chans:, :, :]

        recon_t_mu = recon_t[:, 0:num_half_chans, :, :]
        recon_t_logvar = recon_t[:, num_half_chans:, :, :]

        if GAUSSIAN_VAR_BREAK:
            recon_s_std = ones_like(recon_s_mu) * DECODER_STD
            recon_t_std = ones_like(recon_t_mu) * DECODER_STD
        else:

            recon_s_std = F.softplus(recon_s_logvar) / 0.6931 + EPS_STD
            recon_t_std = F.softplus(recon_t_logvar) / 0.6931 + EPS_STD

        kl = kl_gaussian(recon_t_mu, recon_t_std, recon_s_mu, recon_s_std)

    else:
        raise NotImplementedError('bug')

    return kl


def log_logistic_256(x, mean, logvar, average=False, reduce=True, dim=None):
    ''' from jmtomczak's github'''
    bin_size, scale = 1. / 256., torch.exp(logvar)
    x = (torch.floor(x / bin_size) * bin_size - mean) / scale
    cdf_plus = F.sigmoid(x + bin_size / scale)
    cdf_minus = F.sigmoid(x)

    # calculate final log-likelihood for an image
    log_logist_256 = -torch.log(cdf_plus - cdf_minus + 1.e-7)

    if reduce:
        reduction_fn = torch.mean if average else torch.sum
        return reduction_fn(log_logist_256, dim)

    return log_logist_256


def nll_activation(logits, nll_type, binarize=False):
    ''' helper to activate logits based on the NLL '''
    if nll_type == "clamp":
        num_half_chans = logits.size(1) // 2
        logits_mu = logits[:, 0:num_half_chans, :, :]
        return torch.clamp(logits_mu, min=0. + 1. / 512., max=1. - 1. / 512.)
    elif nll_type == "gaussian":
        num_half_chans = logits.size(1) // 2
        logits_mu = logits[:, 0:num_half_chans, :, :]
        #return F.sigmoid(logits_mu)
        return logits_mu
    elif nll_type == "bernoulli":
        if not binarize:
            return F.sigmoid(logits)
        else:
            return logits > 0

    else:
        raise Exception("unknown nll provided")


def nll(x, recon_x, nll_type):
    ''' helper to get the actual NLL evaluation '''
    nll_map = {
        "gaussian": nll_gaussian,
        "bernoulli": nll_bernoulli,
        "clamp": nll_clamp
    }
    return nll_map[nll_type](x, recon_x)


def nll_bernoulli(x, recon_x_logits):
    batch_size = x.size(0)
    nll = D.Bernoulli(logits=recon_x_logits.view(batch_size, -1)).log_prob(
        x.view(batch_size, -1))
    return -torch.sum(nll, dim=-1)


def nll_clamp(x, recon):
    ''' log-logistic with clamping '''
    batch_size, num_half_chans = x.size(0), recon.size(1) // 2
    recon_mu = recon[:, 0:num_half_chans, :, :].contiguous()
    recon_logvar = recon[:, num_half_chans:, :, :].contiguous()
    return log_logistic_256(x.view(batch_size, -1),
                            torch.clamp(recon_mu.view(batch_size, -1),
                                        min=0. + 1. / 512.,
                                        max=1. - 1. / 512.),
                            F.hardtanh(recon_logvar.view(batch_size, -1),
                                       min_val=-4.5,
                                       max_val=0),
                            dim=-1)


def nll_laplace(x, recon):
    batch_size, num_half_chans = x.size(0), recon.size(1) // 2
    recon_mu = recon[:, 0:num_half_chans, :, :].contiguous()
    recon_logvar = recon[:, num_half_chans:, :, :].contiguous()

    nll = D.Laplace(
        # recon_mu.view(batch_size, -1),
        recon_mu.view(batch_size, -1),
        # F.hardtanh(recon_logvar.view(batch_size, -1), min_val=-4.5, max_val=0) + 1e-6
        recon_logvar.view(batch_size, -1)).log_prob(x.view(batch_size, -1))
    return -torch.sum(nll, dim=-1)


def nll_gaussian(x, recon):
    batch_size, num_half_chans = x.size(0), recon.size(1) // 2
    recon_mu = recon[:, 0:num_half_chans, :, :].contiguous()
    recon_logvar = recon[:, num_half_chans:, :, :].contiguous()
    recon_std = recon_logvar.mul(0.5).exp_()

    if GAUSSIAN_VAR_BREAK:
        recon_std = ones_like(recon_std) * DECODER_STD

    nll = D.Normal(recon_mu.view(batch_size, -1),
                   recon_std.view(batch_size,
                                  -1)).log_prob(x.view(batch_size, -1))
    return -torch.sum(nll, dim=-1)


def nll_gaussian_mu_std(x, mu, std):
    batch_size = x.size(0)
    nlprob = D.Normal(mu.view(batch_size, -1),
                      std.view(batch_size,
                               -1)).log_prob(x.view(batch_size, -1))
    return -torch.sum(nlprob, dim=-1)


def nll_gaussian_N_0_1(x):
    batch_size = x.size(0)
    x = x.view(batch_size, -1)
    nlprob = D.Normal(torch.zeros_like(x), torch.ones_like(x)).log_prob(x)
    return -torch.sum(nlprob, dim=-1)


def prob_ratio_gaussian(z, mu_d, std_d, mu_n, std_n):
    batch_size = mu_d.size(0)

    z = z.view(batch_size, -1)
    mu_d = mu_d.view(batch_size, -1)
    std_d = std_d.view(batch_size, -1)
    mu_n = mu_n.view(batch_size, -1)
    std_n = std_n.view(batch_size, -1)

    log_gaussian_d = -(z - mu_d)**2 / (2 * std_d**2)
    log_gaussian_n = -(z - mu_n)**2 / (2 * std_n**2)

    return torch.sum(log_gaussian_d - log_gaussian_n, dim=-1).exp()
