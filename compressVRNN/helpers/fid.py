import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.model_zoo as model_zoo

from torch.autograd import Variable
from copy import deepcopy
from collections import Counter
from torchvision.models.inception import InceptionA, InceptionB, \
    InceptionC, InceptionD, InceptionE, BasicConv2d, InceptionAux

from optimizers.adamnormgrad import AdamNormGrad
from datasets.loader import get_loader

from .utils import float_type, check_or_create_dir, num_samples_in_loader
from .metrics import softmax_accuracy
from .layers import View, Identity, flatten_layers, EarlyStopping, \
    BWtoRGB
from .resnet_models import resnet18


def build_optimizer(model, args):
    optim_map = {
        "rmsprop": optim.RMSprop,
        "adam": optim.Adam,
        "adamnorm": AdamNormGrad,
        "adadelta": optim.Adadelta,
        "sgd": optim.SGD,
        "lbfgs": optim.LBFGS
    }
    # filt = filter(lambda p: p.requires_grad, model.parameters())
    # return optim_map[args.optimizer.lower().strip()](filt, lr=args.lr)
    return optim_map[args.optimizer.lower().strip()](model.parameters(),
                                                     lr=args.lr)


def train(epoch, model, optimizer, data_loader, args):
    model.train()
    for data, target in data_loader.train_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()

        # project to the output dimension
        output, _ = model(data)
        loss = model.loss_function(output, target)
        correct = softmax_accuracy(output, target)

        # compute loss
        loss.backward()
        optimizer.step()

    num_samples = len(data_loader.train_loader.dataset)

    print(
        '[FID]Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.4f}'
        .format(epoch, num_samples, num_samples,
                100. * num_samples / num_samples, loss.data.item(), correct))


def test(epoch, model, data_loader, args):
    model.eval()
    loss, correct, num_samples = [], [], 0

    for data, target in data_loader.test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():

            output, _ = model(data)
            loss_t = model.loss_function(output, target)
            correct_t = softmax_accuracy(output, target)

            loss.append(loss_t.detach().cpu().item())
            correct.append(correct_t)
            num_samples += data.size(0)

    loss = np.mean(loss)
    acc = np.mean(correct)
    print(
        '\n[FID {} samples]Test Epoch: {}\tAverage loss: {:.4f}\tAverage Accuracy: {:.4f}\n'
        .format(num_samples, epoch, loss, acc))
    return loss, acc


def train_fid_model(args, fid_type='conv', batch_size=32):
    ''' builds and trains a classifier '''
    loader = get_loader(args)

    # debug prints
    print("[FID] train = ",
          num_samples_in_loader(loader.train_loader), " | test = ",
          num_samples_in_loader(loader.test_loader), " | output_classes = ",
          loader.output_size)

    model = FID(loader.img_shp,
                loader.output_size,
                batch_size=batch_size,
                fid_type=fid_type,
                kwargs=vars(args))
    if not model.model_exists:
        optimizer = build_optimizer(model, args)
        early_stop = EarlyStopping(model, max_steps=20)

        for epoch in range(1, args.epochs + 1):
            train(epoch, model, optimizer, loader, args)
            loss, acc = test(epoch, model, loader, args)
            if early_stop(1 - acc):
                early_stop.restore()
                break

    # test one final time to check accuracy .
    # this is useful to validate loaded models
    # Doesn't make sense for pretrained inceptionv3
    if fid_type == 'conv':
        test(epoch=-1, model=model, data_loader=loader, args=args)

    del loader  # force cleanup
    return model


class InceptionV3UptoPool3(nn.Module):
    def __init__(self, num_classes=1000, transform_input=True):
        super(InceptionV3UptoPool3, self).__init__()
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x_pool2d = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x_pool2d, training=self.training, inplace=False)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        return x, x_pool2d


class ConvFID(nn.Module):
    def __init__(self, input_shape, output_size):
        super(ConvFID, self).__init__()
        model = nn.Sequential(
            resnet18(pretrained=True, input_chans=input_shape[0]),
            nn.AdaptiveAvgPool2d((1, 1)), View((-1, 512)),
            nn.Linear(512, output_size))
        self.first_section = model[0:-1]  # extract a feature layer
        self.second_section = model[-1:]

    def forward(self, x):
        features = self.first_section(x)
        return self.second_section(features).squeeze(), features


class FID(nn.Module):
    def __init__(self,
                 input_shape,
                 output_size,
                 batch_size,
                 fid_type='inceptionv3',
                 **kwargs):
        super(FID, self).__init__()
        assert fid_type == 'conv' or 'inceptionv3'

        self.input_shape = input_shape
        self.output_size = output_size
        self.batch_size = batch_size
        self.fid_type = fid_type
        self.is_color = input_shape[0] > 1
        self.chans = 3 if self.is_color else 1

        # grab the meta config
        self.config = kwargs['kwargs']

        # build the encoder and decoder
        self.model = self._build_inception()
        self.model_exists = self.load()

    def _build_inception(self):
        if self.fid_type == 'inceptionv3':
            print("compiling inception_v3 FID model")
            model = nn.Sequential(
                BWtoRGB(),
                nn.Upsample(size=[299, 299], mode='bilinear'),
                InceptionV3UptoPool3()  #self.output_size)
            )
        else:
            print("compiling standard convnet FID model")
            model = ConvFID(self.input_shape, self.output_size)

        # push to multi-gpu
        if self.config['ngpu'] > 1:
            model = nn.DataParallel(model)

        # push to cuda
        if self.config['cuda']:
            model.cuda()

        return model

    def load(self):
        # load the FID model if it exists
        if self.fid_type == 'inceptionv3':
            # load the state dict from the zoo
            model_url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
            self.model[-1].load_state_dict(model_zoo.load_url(model_url))
            print("successfully loaded inceptionv3")
            return True
        else:

            model_filename = self.config['fid_model']
            if os.path.isfile(model_filename):
                print("loading existing FID model")
                self.load_state_dict(torch.load(model_filename))
                return True
            else:
                return False

    def save(self, overwrite=False):
        # save the FID model if it doesnt exist
        model_filename = self.config['fid_model']
        if not os.path.isfile(model_filename) or overwrite:
            print("saving existing FID model")
            torch.save(self.state_dict(), model_filename)

    def get_name(self):
        full_hash_str = "_type{}_input{}_output{}_batch{}_lr{}_ngpu{}".format(
            str(self.fid_type), str(self.input_shape), str(self.output_size),
            str(self.batch_size), str(self.config['lr']),
            str(self.config['ngpu']))

        # cleanup symbols that would cause filename issues
        full_hash_str = full_hash_str.strip().lower().replace('[', '')  \
                                                     .replace(']', '')  \
                                                     .replace(' ', '')  \
                                                     .replace('{', '') \
                                                     .replace('}', '') \
                                                     .replace(',', '_') \
                                                     .replace(':', '') \
                                                     .replace('(', '') \
                                                     .replace(')', '') \
                                                     .replace('\'', '')
        return 'fid_' + FID._clean_task_str(str(
            self.config['task'])) + full_hash_str

    @staticmethod
    def _clean_task_str(task_str):
        ''' helper to reduce string length.
            eg: mnist+svhn+mnist --> mnist2svhn1 '''
        result_str = ''
        if '+' in task_str:
            splits = Counter(task_str.split('+'))
            for k, v in splits.items():
                result_str += '{}{}'.format(k, v)

            return result_str

        return task_str

    def loss_function(self, pred, target):
        return F.cross_entropy(pred, target)

    def forward(self, x):
        output, feat = self.model(x)
        return output, feat
