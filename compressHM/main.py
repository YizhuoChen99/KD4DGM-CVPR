






















import os
import argparse
from numpy import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from models.HM.HM import HM
from models.student_teacher import StudentTeacher
from helpers.utils import number_of_parameters
import GPUs

parser = argparse.ArgumentParser(description='compress HM')

# Task parameters
parser.add_argument('--uid', type=str, default="", help="unique id")
parser.add_argument('--epochs',
                    type=int,
                    default=1000,
                    metavar='N',
                    help='number of epochs')

parser.add_argument('--ckpt-dir',
                    type=str,
                    default='./CKPT',
                    metavar='OD',
                    help='directory which contains ckpt')

# Optimization related
parser.add_argument('--optimizer',
                    type=str,
                    default="adam",
                    help="specify optimizer")
parser.add_argument('--lr',
                    type=float,
                    default=1e-2,
                    metavar='LR',
                    help='learning rate')
parser.add_argument('--batch-size',
                    type=int,
                    default=272,
                    metavar='N',
                    help='input batch size for training')
parser.add_argument('--teacher-model', type=str, default=None, help='help')
parser.add_argument('--eval', action='store_true', default=False, help='help')
parser.add_argument('--resume', type=int, default=0, help='help')
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
parser.add_argument('--gpu-wait', type=float, default=1.0, help='wait until')
parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help='disables CUDA training')
args = parser.parse_args()

Y1_MEAN = 3.48778309
Y2_MEAN = 70.89705882
Y1_STD = 1.13927121
Y2_STD = 13.56996002


class OFDataset(Dataset):
    def __init__(self):
        super().__init__()
        with open('./OF.txt', 'r') as f:
            lines = f.readlines()
        self.data = []
        for line in lines:
            line = line.strip('\n').split()
            y1 = (float(line[1]) - Y1_MEAN) / Y1_STD
            y2 = (float(line[2]) - Y2_MEAN) / Y2_STD

            d = [y1, y2]
            self.data += [d]

        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx >= self.len:
            raise IndexError

        d = self.data[idx]
        y1 = d[0:1]
        y2 = d[1:2]

        return torch.tensor(y1).cuda(), torch.tensor(y2).cuda()


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


def train(epoch, model, gen_optimizer, inf_optimizer, loader):
    ''' train loop helper '''
    return execute_graph(epoch=epoch,
                         model=model,
                         data_loader=loader,
                         gen_optimizer=gen_optimizer,
                         inf_optimizer=inf_optimizer,
                         prefix='train')


def execute_graph(epoch,
                  model,
                  data_loader,
                  gen_optimizer=None,
                  inf_optimizer=None,
                  prefix='test'):
    loss_map, num_samples = {}, 0

    for y1, y2 in data_loader:

        gen_optimizer.zero_grad()

        if args.teacher_model is None:
            loss, loss_z, loss_y = model.wake_student({'y': [y1, y2]})
            loss_map['wake_loss_z_mean'] = loss_z
            loss_map['wake_loss_y_mean'] = loss_y
        else:
            loss = model.distill()

        loss.backward()
        loss_map['gen_param_norm_mean'] = torch.norm(
            nn.utils.parameters_to_vector(model.student.gennet.parameters()))
        loss_map['gen_grad_norm_mean'] = torch.norm(
            parameters_grad_to_vector(model.student.gennet.parameters()))
        loss_map['wake_loss_mean'] = loss

        gen_optimizer.step()

        if args.teacher_model is None:
            inf_optimizer.zero_grad()
            loss = model.sleep_student()

            loss.backward()
            loss_map['inf_param_norm_mean'] = torch.norm(
                nn.utils.parameters_to_vector(
                    model.student.infnet.parameters()))
            loss_map['inf_grad_norm_mean'] = torch.norm(
                parameters_grad_to_vector(model.student.infnet.parameters()))
            loss_map['sleep_loss_mean'] = loss

            inf_optimizer.step()

    print('{}[Epoch {}][{} samples]: losses {}'.format(prefix, epoch,
                                                       num_samples,
                                                       str(loss_map)),
          flush=True)

    loss_map.clear()

    return


def generate(HM, loader, fname):
    # sns.set_theme(style="darkgrid")
    sns.set_style("whitegrid")

    context = sns.plotting_context()
    context.update({
        'xtick.labelsize': 14,
        'axes.labelsize': 14.0,
        'font.size': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.figsize': (5, 4)
    })
    sns.set_context(context)

    y1, y2 = loader.__iter__().__next__()
    y1 = y1.squeeze().cpu().numpy() * Y1_STD + Y1_MEAN
    y2 = y2.squeeze().cpu().numpy() * Y2_STD + Y2_MEAN
    cmap = sns.cubehelix_palette(start=0, light=0.9, as_cmap=True)

    sns.kdeplot(data={
        'duration': y1,
        'waiting': y2
    },
                x='duration',
                y='waiting',
                cmap=cmap,
                fill=True,
                clip=((0.5, 6), (30, 105)),
                levels=10,
                legend=False)

    plt.subplots_adjust(left=0.16, right=0.96, bottom=0.16, top=0.95)

    plt.savefig('data.png')
    plt.cla()

    param = HM.generate()
    y1, y2 = param['y']
    y1 = y1.squeeze().detach().cpu().numpy() * Y1_STD + Y1_MEAN
    y2 = y2.squeeze().detach().cpu().numpy() * Y2_STD + Y2_MEAN

    s = 0

    cmap = sns.cubehelix_palette(start=s, light=0.9, as_cmap=True)

    sns.kdeplot(data={
        'duration': y1,
        'waiting': y2
    },
                x='duration',
                y='waiting',
                cmap=cmap,
                fill=True,
                clip=((0.5, 6), (30, 105)),
                levels=10,
                legend=False)
    plt.savefig(fname + '.png')
    plt.cla()


def get_model_and_loader():
    ''' helper to return the model and the loader '''

    loader = DataLoader(OFDataset(), batch_size=args.batch_size, shuffle=False)

    if args.teacher_model is None:
        student = HM(2, kwargs=vars(args))
        teacher = None
    else:
        student = HM(1, kwargs=vars(args))
        teacher = HM(2, kwargs=vars(args))

    student_teacher = StudentTeacher(teacher, student, kwargs=vars(args))

    if args.teacher_model is not None:
        teacher_fname = os.path.join(args.ckpt_dir, args.teacher_model)
        print("loading teacher model {}".format(teacher_fname), flush=True)
        student_teacher.teacher.load_state_dict(torch.load(teacher_fname),
                                                strict=True)

    return [student_teacher, loader]


def set_seed(seed):
    if seed is None:
        raise NotImplementedError('seed must be specified')
    print("setting seed %d" % args.seed, flush=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUs.select(args.gpu_wait)

    try:
        os.makedirs(args.ckpt_dir)
    except OSError:
        pass

    # handle randomness / non-randomness
    set_seed(args.seed)

    # cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # collect our model and data loader
    model, data_loader = get_model_and_loader()

    # eval mode
    if args.eval:
        fname = args.uid + '-' + str(args.resume)
        generate(model.student, data_loader, fname)

    # train mode
    else:
        gen_optimizer = build_optimizer(model.student.gennet)
        inf_optimizer = build_optimizer(model.student.infnet)
        gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, 99999, 0.1)
        inf_scheduler = optim.lr_scheduler.StepLR(inf_optimizer, 99999, 0.1)

        print(
            "there are {} params with {} elems in the st-model and {} params in the student with {} elems"
            .format(len(list(model.parameters())), number_of_parameters(model),
                    len(list(model.student.gennet.parameters())),
                    number_of_parameters(model.student.gennet)),
            flush=True)

        for epoch in range(args.resume + 1, args.epochs + 1):
            train(epoch, model, gen_optimizer, inf_optimizer, data_loader)
            gen_scheduler.step()
            inf_scheduler.step()
            if epoch % 1000 == 0:
                fname = args.uid
                generate(model.student, data_loader, fname)
                save_fname = os.path.join(args.ckpt_dir,
                                          '{}-model.pth.tar'.format(fname))
                print("saving model {}...".format(save_fname), flush=True)
                torch.save(model.student.state_dict(), save_fname)


if __name__ == "__main__":
    run(args)
