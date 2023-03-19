from __future__ import division
import os
import argparse
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence

from models.vrnn.VRNN import VRNN
from models.student_teacher import StudentTeacher
import GPUs

parser = argparse.ArgumentParser(description='compress VRNN')

# Task parameters
parser.add_argument('--uid', type=str, default="", help="unique id")
parser.add_argument('--epochs',
                    type=int,
                    default=200,
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
                    default=1e-4,
                    metavar='LR',
                    help='learning rate')
parser.add_argument('--batch-size',
                    type=int,
                    default=32,
                    metavar='N',
                    help='input batch size for training')
parser.add_argument('--dzlambda', type=float, default=1e-3, help='dzlambda')
parser.add_argument('--STD', type=float, default=1, metavar='LR', help='STD')

parser.add_argument('--teacher-model', type=str, default=None, help='help')

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


def fetch_iamondb():

    train_npy_x = "train_npy_x.npy"

    train_x = pickle.load(open(train_npy_x, mode="rb"))

    return train_x


class IAMDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.x = fetch_iamondb()

        self.len = len(self.x)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx >= self.len:
            raise IndexError

        x = self.x[idx]

        return torch.tensor(x).cuda()


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


def train(epoch, model, optimizer, loader):
    ''' train loop helper '''
    return execute_graph(epoch=epoch,
                         model=model,
                         data_loader=loader,
                         optimizer=optimizer,
                         prefix='train')


def execute_graph(epoch, model, data_loader, optimizer=None, prefix='train'):

    loss_t, loss_map, num_samples = {}, {}, 0

    repeat = 1

    for i, (x, lengths) in enumerate(data_loader):

        model.student.train()

        if i % repeat == 0:
            optimizer.zero_grad()

        if args.teacher_model is not None:
            loss, loss_t = model.distill()
        else:
            loss, loss_t = model.forward_student(x, lengths)

        loss_repeat = loss / repeat
        loss_repeat.backward()

        if i % repeat == repeat - 1:
            loss_t['loss_mean'] = loss
            loss_t['param_norm_mean'] = torch.norm(
                nn.utils.parameters_to_vector(model.student.parameters()))
            loss_t['grad_norm_mean'] = torch.norm(
                parameters_grad_to_vector(model.student.parameters()))
            optimizer.step()

            loss_map = _add_loss_map(loss_map, loss_t)
            num_samples += x.shape[1] * repeat

            print('{}[Epoch {}][{} samples]: loss {}'.format(
                prefix, epoch, num_samples, loss.detach()),
                  flush=True)

            if num_samples % 320 == 0:
                fname = './IMAGE/{}-{}-{}'.format(args.uid, epoch, num_samples)
                generate(model.student, fname)

    loss_map = _mean_map(loss_map)  # reduce the map to get actual means
    print('{}[Epoch {}][{} samples]: losses {}'.format(prefix, epoch,
                                                       num_samples,
                                                       str(loss_map)),
          flush=True)

    loss_map.clear()

    return


def plot_lines_iamondb_example(X, x0):

    X_mean = np.array([8.17868533, -0.11164117])
    X_std = np.array([41.95389001, 37.123557])

    X = np.concatenate([x0, X], axis=0)

    for i in range(1, X.shape[0]):
        X[i, 1:] = X[i - 1, 1:] + X[i, 1:] * X_std + X_mean

    non_contiguous = np.where(X[:, 0] == 1)[0] + 1
    # places where new stroke begin
    start = 0

    for end in non_contiguous:
        plt.plot(X[start:end, 1], X[start:end, 2])
        start = end

    plt.plot(X[start:, 1], X[start:, 2])


def generate(VRNN, fname):
    VRNN.eval()

    param = VRNN.generate(1.0, 0.0, 1.0)
    lines = param['x'].detach().cpu().numpy()

    for index in range(0, 28, 5):
        plt.figure(figsize=(5, 4))
        plt.axis('equal')
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
        for i in range(5):
            plot_lines_iamondb_example(lines[:, i + index, :],
                                       np.array([[0, 500, 1000 * i + 500]]))
        plt.savefig('{}-{}.png'.format(fname, index))
        plt.cla()


def get_model_and_loader():
    ''' helper to return the model and the loader '''

    dataset = IAMDataset()

    def collate_fn(train_data):
        train_data.sort(key=lambda data: len(data), reverse=True)
        data_length = [len(data) for data in train_data]
        train_data = pad_sequence(train_data,
                                  batch_first=False,
                                  padding_value=0)
        return train_data, data_length

    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        drop_last=True,
                        collate_fn=collate_fn)

    if args.teacher_model is not None:
        student = VRNN(1, kwargs=vars(args))
        teacher = VRNN(2, kwargs=vars(args))
    else:
        student = VRNN(2, kwargs=vars(args))
        teacher = None

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

    # handle randomness / non-randomness
    set_seed(args.seed)

    try:
        os.makedirs(args.ckpt_dir)
    except OSError:
        pass

    try:
        os.makedirs('./IMAGE')
    except OSError:
        pass

    # cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # collect our model and data loader
    model, data_loader = get_model_and_loader()

    optimizer = build_optimizer(model.student)

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, optimizer, data_loader)

        save_fname = os.path.join(args.ckpt_dir,
                                  '{}-model.pth.tar'.format(args.uid, epoch))
        torch.save(model.student.state_dict(), save_fname)


if __name__ == "__main__":
    run(args)
