import os
import argparse
import pprint
from numpy import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.vae.LVAE import LVAE
from models.student_teacher import StudentTeacher
from datasets.loader import get_loader
from helpers.grapher import Grapher
from helpers.utils import num_samples_in_loader
from helpers.distributions import set_decoder_std
from helpers.metrics_image import compute_score_raw
import GPUs

parser = argparse.ArgumentParser(description='compress VAE')

# Task parameters
parser.add_argument('--uid', type=str, default="", help="unique id")
parser.add_argument('--task',
                    type=str,
                    default="celeba",
                    help="task to work on")
parser.add_argument('--epochs',
                    type=int,
                    default=50,
                    metavar='N',
                    help='number of epochs')

parser.add_argument('--download', type=int, default=1, help='download dataset')
parser.add_argument('--data-dir',
                    type=str,
                    default='../datasets-torchvision',
                    metavar='DD',
                    help='directory which contains input data')
parser.add_argument('--ckpt-dir',
                    type=str,
                    default='./CKPT',
                    metavar='OD',
                    help='directory which contains ckpt')

# train / eval

parser.add_argument('--teacher-model',
                    type=str,
                    default=None,
                    help='teacher you are going to learn from')
parser.add_argument('--eval-model',
                    type=str,
                    default=None,
                    help='model you are to eval')

# Model parameters
parser.add_argument('--nll-type', type=str, default='gaussian', help='')
parser.add_argument('--t-arch', type=int, default=1, help='teacher arch')
parser.add_argument('--s-arch', type=int, default=1, help='student arch')
# Optimization related
parser.add_argument('--optimizer',
                    type=str,
                    default="adam",
                    help="specify optimizer")
parser.add_argument('--lr',
                    type=float,
                    default=1e-4,
                    metavar='LR',
                    help='learning rate')
parser.add_argument('--batch-size',
                    type=int,
                    default=128,
                    metavar='N',
                    help='input batch size for training')

parser.add_argument('--beta',
                    type=float,
                    default=1e-1,
                    help='hyperparameter to scale KL term in ELBO')

parser.add_argument('--warmup-epoch',
                    type=int,
                    default=25,
                    help='warmup epoch')

parser.add_argument('--distill-z-kl-lambda',
                    type=float,
                    default=1e-1,
                    help='distill z kl lambda')
parser.add_argument('--distill-z-reduction',
                    type=str,
                    default='mean',
                    help='how to reduce z')

# Visdom parameters
parser.add_argument('--visdom-url',
                    type=str,
                    default="http://localhost",
                    help='visdom URL for graphs (default: http://localhost)')
parser.add_argument('--visdom-port',
                    type=int,
                    default="8097",
                    help='visdom port for graphs (default: 8097)')

# Device parameters
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    help='seed for numpy and pytorch')
parser.add_argument('--ngpu',
                    type=int,
                    default=1,
                    help='number of gpus available')
parser.add_argument('--gpu-wait', type=float, default=1.0, help='wait until')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='disables CUDA training')
args = parser.parse_args()


def parameters_grad_to_vector(parameters) -> torch.Tensor:

    vec = []
    for param in parameters:
        if param.grad is not None:
            vec.append(param.grad.view(-1))
    return torch.cat(vec)


def build_optimizer(model):
    optim_map = {
        "rmsprop": optim.RMSprop,
        "adam": optim.Adam,
        "adadelta": optim.Adadelta,
        "sgd": optim.SGD,
        "lbfgs": optim.LBFGS
    }

    return optim_map[args.optimizer.lower().strip()](model.parameters(),
                                                     lr=args.lr,
                                                     weight_decay=1e-4)


def register_plots(loss, grapher, epoch, prefix='train'):
    for k, v in loss.items():
        if isinstance(v, map):
            register_plots(loss[k], grapher, epoch, prefix=prefix)

        if 'mean' in k:
            key_name = k
            value = v.item() if not isinstance(v, (float, np.float32,
                                                   np.float64)) else v
            grapher.register_single(
                {'%s_%s' % (prefix, key_name): [[epoch], [value]]},
                plot_type='line')


def register_images(images, names, grapher, prefix="train"):
    ''' helper to register a list of images '''
    if isinstance(images, list):
        assert len(images) == len(names)
        for im, name in zip(images, names):
            register_images(im, name, grapher, prefix=prefix)
    else:
        images = torch.clamp(images.detach(), 0.0, 1.0)
        grapher.register_single({'{}_{}'.format(prefix, names): images},
                                plot_type='imgs')


def _add_loss_map(loss_tm1, loss_t):
    if not loss_tm1:  # base case: empty dict
        resultant = {'count': 1}
        for k, v in loss_t.items():
            if 'mean' in k:
                resultant[k] = v.detach()

        return resultant

    resultant = {}
    for (k, v) in loss_t.items():
        if 'mean' in k:
            resultant[k] = loss_tm1[k] + v.detach()

    # increment total count
    resultant['count'] = loss_tm1['count'] + 1
    return resultant


def _mean_map(loss_map):
    for k in loss_map.keys():
        loss_map[k] /= loss_map['count']

    return loss_map


def train(epoch, model, optimizer, loader, grapher):
    return execute_graph(epoch=epoch,
                         model=model,
                         data_loader=loader,
                         grapher=grapher,
                         optimizer=optimizer)


def test(model, loader):
    model.eval()

    score = compute_score_raw(dataset=loader,
                              sampleSize=1024,
                              batchSize=64,
                              saveFolder_r=f'./temp_images/{args.uid}/real/',
                              saveFolder_f=f'./temp_images/{args.uid}/fake/',
                              netG=model.student,
                              conv_model='inception_v3',
                              workers=4)

    print('Evaluation scores:', flush=True)

    score_names = ['emd', 'mmd', '1nn', 'FID']

    for i in range(len(score)):
        print(f'{score_names[i]}: {score[i]}', flush=True)

    return


def execute_graph(epoch, model, data_loader, grapher, optimizer):

    model.student.train()

    loss_map, num_samples = {}, 0

    for data, _ in data_loader:
        optimizer.zero_grad()
        beta = min((epoch / args.warmup_epoch), 1.0) * args.beta

        if model.teacher is None:
            data = data.cuda() if args.cuda else data
            output_map = model(data)

        else:
            output_map = model.distill()

        loss_t = model.loss_function(output_map, beta)

        loss_t['loss_mean'].backward()
        loss_t['param_norm_mean'] = torch.norm(
            nn.utils.parameters_to_vector(model.student.parameters()))
        loss_t['grad_norm_mean'] = torch.norm(
            parameters_grad_to_vector(model.student.parameters()))
        optimizer.step()

        loss_map = _add_loss_map(loss_map, loss_t)
        num_samples += data.size(0)

    loss_map = _mean_map(loss_map)  # reduce the map to get actual means
    print('[Epoch {}][{} samples]: losses {}'.format(epoch, num_samples,
                                                     str(loss_map)),
          flush=True)
    if grapher:  # only if grapher is not None
        register_plots(loss_map, grapher, epoch=epoch, prefix='train')

        images = []
        img_names = []
        if 'student' in output_map.keys() and 'x' in output_map.keys():
            images += [output_map['x'], output_map['student']['x_reconstr']]
            img_names += ['original_imgs', 'vae_reconstructions']
        if 'distill' in output_map.keys():
            images += [
                output_map['distill']['gen_teacher'],
                output_map['distill']['gen_student']
            ]
            img_names += ['generation_teacher', 'generation_student']
        register_images(images, img_names, grapher, prefix='train')
        generate(model.student, grapher)  # generate samples
        grapher.show()

    loss_map.clear()

    return


def generate(vae, grapher):

    vae.eval()
    _, gen, _ = vae.generate_synthetic_samples(args.batch_size)
    gen = torch.clamp(gen, 0, 1)
    grapher.register_single({'generated_samples': gen}, plot_type='imgs')
    grapher.show()


def get_model_and_loader():
    ''' helper to return the model and the loader '''

    loader = get_loader(args)

    print("train = ",
          num_samples_in_loader(loader.train_loader),
          " | test = ",
          num_samples_in_loader(loader.test_loader),
          " | shape ",
          loader.img_shp,
          flush=True)

    # append the image shape to the config & build the VAE
    args.img_shp = loader.img_shp,

    s_vae = LVAE(loader.img_shp, args.s_arch, kwargs=vars(args))

    t_vae = None

    if args.eval_model is not None:
        print("loading eval model {}".format(args.eval_model), flush=True)
        s_vae.load_state_dict(torch.load(args.eval_model), strict=True)

    # train mode
    else:
        # with a teacher
        if args.teacher_model is not None:
            t_vae = LVAE(loader.img_shp, args.t_arch, kwargs=vars(args))
            print("loading teacher model {}".format(args.teacher_model),
                  flush=True)

            # init t_vae
            t_vae.load_state_dict(torch.load(args.teacher_model), strict=True)
            t_vae.requires_grad_(False)
            s_vae.u1.requires_grad_(False)
            s_vae.u2.requires_grad_(False)
            s_vae.u3.requires_grad_(False)
            s_vae.u4.requires_grad_(False)
            s_vae.u5.requires_grad_(False)

    # build the combiner which takes in the VAE as a parameter
    # and projects the latent representation to the output space
    student_teacher = StudentTeacher(t_vae, s_vae, kwargs=vars(args))

    # build the grapher object
    grapher = Grapher(env=args.uid,
                      server=args.visdom_url,
                      port=args.visdom_port)

    return [student_teacher, loader, grapher]


def set_seed(seed):
    if seed is None:
        raise NotImplementedError('seed must be specified')
    print("setting seed %d" % args.seed, flush=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True


def transfer_beta_sigma():
    sigma = np.sqrt(args.beta / 2)
    args.beta = 1.0
    set_decoder_std(sigma)


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUs.select(args.gpu_wait)

    # handle randomness / non-randomness
    set_seed(args.seed)

    try:
        os.makedirs(args.ckpt_dir)
    except OSError:
        pass

    transfer_beta_sigma()

    # cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # collect our model and data loader
    model, data_loader, grapher = get_model_and_loader()

    # eval mode
    if args.eval_model is not None:
        print("evaluating model {}...".format(args.eval_model), flush=True)
        test(model, data_loader.test_loader)

    # train mode
    else:
        optimizer = build_optimizer(model.student)  # collect our optimizer

        for epoch in range(1, args.epochs + 1):
            train(epoch, model, optimizer, data_loader.train_loader, grapher)

        test(model, data_loader.test_loader)

        save_fname = os.path.join(args.ckpt_dir, args.uid + '-model.pth.tar')
        print("saving model {}...".format(save_fname), flush=True)
        torch.save(model.student.state_dict(), save_fname)

    # dump config to visdom
    grapher.vis.text(pprint.PrettyPrinter(indent=4).pformat(
        model.student.config),
                     opts=dict(title="config"))

    grapher.save()  # save the remote visdom graphs


if __name__ == "__main__":
    run(args)
