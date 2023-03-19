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

# ensures we get the same permutation
PERMUTE_SEED = 1

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


def get_samplers(num_classes):
    ''' builds samplers taking into account previous classes'''
    # NOTE: test datasets are now merged via sequential_test_set_merger

    test_samplers = [
        ClassSampler(class_number=j, shuffle=False) for j in range(num_classes)
    ]
    train_samplers = [
        ClassSampler(class_number=j, shuffle=True) for j in range(num_classes)
    ]
    return train_samplers, test_samplers


def get_loader(args,
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

    if task == 'mnist':
        transform = [transforms.Resize((32, 32), PIL.Image.NEAREST)]

    assert task in loader_map, "unknown task requested"

    return loader_map[task](path=data_dir,
                            batch_size=args.batch_size,
                            transform=transform,
                            target_transform=target_transform,
                            train_sampler=train_sampler,
                            test_sampler=test_sampler,
                            use_cuda=args.cuda)
