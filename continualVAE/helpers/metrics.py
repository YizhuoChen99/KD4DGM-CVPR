import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as D

from scipy import linalg
from torch.autograd import Variable

from .utils import to_data, num_samples_in_loader


def softmax_correct(preds, targets):
    pred = to_data(preds).max(1)[1]  # get the index of the max log-probability
    targ = to_data(targets)
    return pred.eq(targ).cpu().type(torch.FloatTensor)


def softmax_accuracy(preds, targets, size_average=True):
    reduction_fn = torch.mean if size_average is True else torch.sum
    return reduction_fn(softmax_correct(preds, targets))


def frechet_gauss_gauss_np(synthetic_features, test_features, eps=1e-6):
    # calculate the statistics required for frechet distance
    # https://github.com/bioinf-jku/TTUR/blob/master/fid.py
    mu_synthetic = np.mean(synthetic_features, axis=0)
    sigma_synthetic = np.cov(synthetic_features, rowvar=False)
    mu_test = np.mean(test_features, axis=0)
    sigma_test = np.cov(test_features, rowvar=False)

    diff = mu_synthetic - mu_test

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma_synthetic.dot(sigma_test), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_synthetic.shape[0]) * eps
        covmean = linalg.sqrtm(
            (sigma_synthetic + offset).dot(sigma_test + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-2):
            offset = np.eye(sigma_synthetic.shape[0]) * eps
            covmean = linalg.sqrtm(
                (sigma_synthetic + offset).dot(sigma_test + offset))
            print('offset added', flush=True)

        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma_synthetic) + np.trace(
        sigma_test) - 2 * tr_covmean
    # m = np.square(mu_synthetic - mu_test).sum()
    # s = sp.linalg.sqrtm(np.dot(sigma_synthetic, sigma_test))
    # dist = m + np.trace(sigma_synthetic + sigma_synthetic - 2*s)
    # if np.isnan(dist):
    #     raise Exception("nan occured in FID calculation.")

    # return dist


def frechet_gauss_gauss(dist_a, dist_b):
    ''' d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)). '''
    m = torch.pow(dist_a.loc - dist_b.loc, 2).sum()
    s = torch.sqrt(dist_a.scale * dist_b.scale)
    return torch.mean(m + dist_a.scale + dist_b.scale - 2 * s)


def calculate_fid(fid_model,
                  model,
                  continual_step,
                  loader,
                  grapher,
                  num_samples,
                  cuda=False):
    # evaluate and cache away the FID score
    # model batch size can be larger than fid
    # in order to evaluate large models like inceptionv3
    fid = calculate_fid_from_generated_images(fid_model=fid_model,
                                              vae=model,
                                              continual_step=continual_step,
                                              data_loader=loader,
                                              num_samples=num_samples,
                                              cuda=cuda)
    if grapher:
        grapher.vis.text(str(fid), opts=dict(title="FID"))

    return fid


def calculate_fid_from_generated_images(fid_model,
                                        vae,
                                        continual_step,
                                        data_loader,
                                        num_samples=2000,
                                        cuda=False):
    ''' Extract features and computes the FID score for the VAE vs. the classifier
        NOTE: expects a trained fid classifier and model '''
    fid_model.eval()
    vae.eval()

    # override the min sample count by comparing against the test set
    num_test_samples = num_samples_in_loader(data_loader)
    num_samples = min(num_samples, num_test_samples)

    print("calculating FID, this can take a while...")
    fid, samples_seen = [], 0
    with torch.no_grad():
        # NOTE: requires shuffled test-data as we pick first num_samples
        for data, _ in data_loader:
            if samples_seen > num_samples:
                continue
            data = data.cuda() if cuda else data
            batch_size = data.size(0)
            _, generated, _ = vae.generate_synthetic_samples(
                batch_size, continual_step)

            _, test_features = fid_model(data)
            _, generated_features = fid_model(generated)
            test_features = test_features.view(fid_model.batch_size, -1)
            generated_features = generated_features.view(
                fid_model.batch_size, -1)
            fid_batch = frechet_gauss_gauss_np(
                generated_features.cpu().numpy(),
                test_features.cpu().numpy())
            fid.append(fid_batch)
            samples_seen += batch_size
            print(samples_seen, fid_batch, flush=True)

    frechet_dist = np.mean(fid)
    print("frechet distance [ {} samples ]: {}\n".format(
        samples_seen, frechet_dist),
          flush=True)
    return frechet_dist
