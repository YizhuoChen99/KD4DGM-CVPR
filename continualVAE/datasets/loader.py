import os
import PIL
import numpy as np
import torchvision.transforms as transforms
from copy import deepcopy

from datasets.class_sampler import ClassSampler
from datasets.cifar import CIFAR10Loader
from datasets.fashion_mnist import FashionMNISTLoader
from datasets.mnist import MNISTLoader
from datasets.omniglot import OmniglotLoader
from datasets.permuted_mnist import PermutedMNISTLoader
from datasets.svhn import SVHNCenteredLoader
from datasets.celeba import CELEBALoader
from datasets.utils import sequential_test_set_merger

loader_map = {
    'mnist': MNISTLoader,
    'omniglot': OmniglotLoader,
    'permuted': PermutedMNISTLoader,
    'fashion': FashionMNISTLoader,
    'cifar10': CIFAR10Loader,
    'svhn': SVHNCenteredLoader,
    'svhn_centered': SVHNCenteredLoader,
    'celeba': CELEBALoader
}


def get_samplers(target_class, reverse):
    ''' builds samplers taking into account previous classes'''
    # NOTE: test datasets are now merged via sequential_test_set_merger
    # test_samplers = [lambda x, j=j: ClassSampler(x, class_number=j)
    #                  for j in range(num_classes)]
    # train_samplers = [lambda x, j=j: ClassSampler(x, class_number=j)
    #                   for j in range(num_classes)]
    test_samplers = [
        ClassSampler(target_class, reverse, j, shuffle=False) for j in range(2)
    ]
    train_samplers = [
        ClassSampler(target_class, reverse, j, shuffle=True) for j in range(2)
    ]
    return train_samplers, test_samplers


def get_loader(args,
               train_batch_size,
               test_batch_size,
               transform=None,
               target_transform=None,
               train_sampler=None,
               test_sampler=None):
    ''' increment_seed: increases permutation rng seed,
        sequentially_merge_test: merge all the test sets sequentially '''
    task = args.task

    # overwrite data dir for fashion MNIST because it has issues being
    # in the same directory as regular MNIST
    if task == 'fashion':
        data_dir = os.path.join(args.data_dir, "fashion")
    else:
        data_dir = args.data_dir

    if task == 'fashion':
        transform = [transforms.Resize((32, 32), PIL.Image.NEAREST)]

    assert task in loader_map, "unknown task requested"

    return loader_map[task](path=data_dir,
                            train_batch_size=train_batch_size,
                            test_batch_size=test_batch_size,
                            transform=transform,
                            target_transform=target_transform,
                            train_sampler=train_sampler,
                            test_sampler=test_sampler,
                            use_cuda=args.cuda)


def get_split_data_loaders(args,
                           target_class,
                           transform=None,
                           target_transform=None):
    ''' helper to return the model and the loader '''
    # we build 10 samplers as all of the below have 10 classes
    train_samplers, test_samplers = get_samplers(target_class, args.reverse)

    if args.eval_model is not None:
        train_samplers = [None, None]

    loaders = []
    for class_index, tr, te in zip(range(2), train_samplers, test_samplers):
        train_batch_size = args.batch_size
        test_batch_size = args.batch_size
        loaders += [
            get_loader(args,
                       train_batch_size=train_batch_size,
                       test_batch_size=test_batch_size,
                       transform=transform,
                       target_transform=target_transform,
                       train_sampler=tr,
                       test_sampler=te)
        ]

    return loaders  # sequential_test_set_merger(loaders)
