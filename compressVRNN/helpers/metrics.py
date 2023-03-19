import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as D

from scipy import linalg
from torch.autograd import Variable

from .utils import to_data, float_type, \
    num_samples_in_loader, zero_pad_smaller_cat, zeros_like


def softmax_correct(preds, targets):
    pred = to_data(preds).max(1)[1]  # get the index of the max log-probability
    targ = to_data(targets)
    return pred.eq(targ).cpu().type(torch.FloatTensor)


def softmax_accuracy(preds, targets, size_average=True):
    reduction_fn = torch.mean if size_average is True else torch.sum
    return reduction_fn(softmax_correct(preds, targets))


def bce_accuracy(pred_logits, targets, size_average=True):
    cuda = is_cuda(pred_logits)
    pred = torch.round(F.sigmoid(to_data(pred_logits)))
    pred = pred.type(int_type(cuda))
    reduction_fn = torch.mean if size_average is True else torch.sum
    return reduction_fn(
        pred.data.eq(to_data(targets)).cpu().type(torch.FloatTensor), -1)


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


def calculate_fid(fid_model, model, loader, grapher, num_samples, cuda=False):
    # evaluate and cache away the FID score
    # model batch size can be larger than fid
    # in order to evaluate large models like inceptionv3
    fid = calculate_fid_from_generated_images(fid_model=fid_model,
                                              vae=model,
                                              data_loader=loader,
                                              num_samples=num_samples,
                                              cuda=cuda)
    if grapher:
        grapher.vis.text(str(fid), opts=dict(title="FID"))

    return fid


def calculate_fid_from_generated_images(fid_model,
                                        vae,
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
            assert fid_model.batch_size <= batch_size
            _, generated, _ = vae.generate_synthetic_samples(batch_size)

            for begin, end in zip(
                    range(0, batch_size, fid_model.batch_size),
                    range(fid_model.batch_size, batch_size + 1,
                          fid_model.batch_size)):
                _, test_features = fid_model(data[begin:end])
                _, generated_features = fid_model(generated[begin:end])
                test_features = test_features.view(fid_model.batch_size, -1)
                generated_features = generated_features.view(
                    fid_model.batch_size, -1)
                fid_batch = frechet_gauss_gauss_np(
                    generated_features.cpu().numpy(),
                    test_features.cpu().numpy())
                fid.append(fid_batch)
                samples_seen += fid_model.batch_size
                print(samples_seen, fid_batch, flush=True)

    frechet_dist = np.mean(fid)
    print("frechet distance [ {} samples ]: {}\n".format(
        samples_seen, frechet_dist),
          flush=True)
    return frechet_dist


def calculate_fid_from_generated_images_gpu(fid_model,
                                            model,
                                            data_loader,
                                            batch_size,
                                            fid_layer_index=-4,
                                            cuda=False):
    ''' Extract features and computes the FID score for the VAE vs. the classifier
        NOTE: expects a trained fid classifier and model '''
    fid_submodel = fid_model[0:fid_layer_index]
    fid_submodel.eval()
    model.eval()

    # calculate how many synthetic images from the student model
    num_test_samples = num_samples_in_loader(data_loader.test_loader)
    num_synthetic = int(np.ceil(num_test_samples // batch_size))
    fid, count = 0.0, 0

    with torch.no_grad():
        synthetic = [
            model.generate_synthetic_samples(model.student, batch_size)
            for _ in range(num_synthetic + 1)
        ]
        for (data, _), generated in zip(data_loader.test_loader, synthetic):
            data = Variable(data).cuda() if cuda else Variable(data)
            fid += frechet_gauss_gauss(
                D.Normal(torch.mean(fid_submodel(generated), dim=0),
                         torch.var(fid_submodel(generated), dim=0)),
                D.Normal(torch.mean(fid_submodel(data), dim=0),
                         torch.var(fid_submodel(data), dim=0))).cpu().numpy()
            count += 1

    frechet_dist = fid / count
    print("frechet distance [ {} samples ]: {}\n".format(
        (num_test_samples // batch_size) * batch_size, frechet_dist))
    return frechet_dist


def calculate_consistency(model, loader, reparam_type, vae_type, cuda=False):
    ''' \sum z_d(teacher) == z_d(student) for all test samples '''
    consistency = 0.0

    if model.current_model > 0 and (reparam_type == 'mixture'
                                    or reparam_type == 'discrete'):
        model.eval()  # prevents data augmentation
        consistency, samples_seen = [], 0

        for img, _ in loader.test_loader:
            with torch.no_grad():
                img = Variable(img).cuda() if cuda else Variable(img)

                output_map = model(img)
                if vae_type == 'parallel':
                    teacher_posterior = output_map['teacher']['params'][
                        'discrete']['logits']
                    student_posterior = output_map['student']['params'][
                        'discrete']['logits']
                elif vae_type == 'sequential':
                    if 'discrete' not in output_map['teacher']['params'][
                            'params_0']:
                        # we need the first reparam to be discrete to calculate
                        # the consistency metric
                        return consistency

                    teacher_posterior = output_map['teacher']['params'][
                        'params_0']['discrete']['logits']
                    student_posterior = output_map['student']['params'][
                        'params_0']['discrete']['logits']
                else:
                    raise Exception('unknown VAE consistency requested')

                teacher_posterior = F.softmax(teacher_posterior, dim=-1)
                student_posterior = F.softmax(student_posterior, dim=-1)
                teacher_posterior, student_posterior \
                    = zero_pad_smaller_cat(teacher_posterior, student_posterior)

                correct = to_data(teacher_posterior).max(1)[1] \
                          == to_data(student_posterior).max(1)[1]
                consistency.append(torch.mean(correct.type(float_type(cuda))))
                # print("teacher = ", teacher_posterior)
                # print("student = ", student_posterior)
                # print("consistency[-1]=", correct)
                samples_seen += img.size(0)

        num_test_samples = num_samples_in_loader(loader.test_loader)
        consistency = np.mean(consistency)
        print("Consistency [#samples: {}]: {}\n".format(
            samples_seen, consistency))
    return np.asarray([consistency])


def estimate_fisher(model,
                    data_loader,
                    batch_size,
                    sample_size=10000,
                    cuda=False):
    model.eval()  # lock BN / dropout, etc
    diag_fisher = {
        k: zeros_like(param)
        for (k, param) in model.named_parameters()
    }
    #num_samples_in_dataset = num_samples_in_loader(data_loader.train_loader)
    num_observed_samples = 0

    for x, _ in data_loader.train_loader:
        model.zero_grad()
        x = Variable(x).cuda() if cuda else Variable(x)
        reconstr_x, params = model(x)
        loss = model.loss_function(reconstr_x, x, params)
        for i in range(x.size(0)):
            model.zero_grad()
            loss['loss'][i].backward(retain_graph=True)
            for k, v in model.named_parameters():
                diag_fisher[k] += v.grad.data**2  #/ num_samples_in_dataset

            num_observed_samples += 1
            if num_observed_samples > sample_size:
                break

    for k in diag_fisher.keys():
        diag_fisher[k] /= num_observed_samples
        diag_fisher[k].requires_grad = False

    return diag_fisher
