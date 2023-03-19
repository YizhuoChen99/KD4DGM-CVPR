import torch
import numpy as np
from torchvision import datasets, transforms

from datasets.utils import permute_lambda, create_loader


def generate_permutation(seed):
    orig_seed = np.random.get_state()
    np.random.seed(seed)
    # print("USING SEED ", seed)
    perms = np.random.permutation(28*28)
    np.random.set_state(orig_seed)
    return perms


class PermutedMNISTLoader(object):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1, **kwargs):
        # generate the unique permutation for this loader
        seed = np.random.randint(1, 9999) if 'seed' not in kwargs else kwargs['seed']
        perm = generate_permutation(seed)
        perm_transform = PermutedMNISTLoader._get_permutation_lambda(perm)
        if transform is not None:
            if isinstance(transform, list):
                transform.insert(0, perm_transform)
        else:
            transform = [perm_transform]

        # get the datasets
        train_dataset, test_dataset = self.get_datasets(path, transform, target_transform)

        # build the loaders
        kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
        self.train_loader = create_loader(train_dataset,
                                          train_sampler,
                                          batch_size,
                                          shuffle=True if train_sampler is None else False,
                                          **kwargs)

        self.test_loader = create_loader(test_dataset,
                                         test_sampler,
                                         batch_size,
                                         shuffle=False,
                                         **kwargs)

        self.output_size = 10
        self.batch_size = batch_size

        # grab a test sample to get the size
        test_img, _ = self.train_loader.__iter__().__next__()
        #print('test img = ', test_img.shape)
        self.img_shp = list(test_img.size()[1:])
        #print("derived image shape = ", self.img_shp)

    @staticmethod
    def _get_permutation_lambda(pixel_permutation):
        return transforms.Lambda(lambda x: permute_lambda(x, pixel_permutation=pixel_permutation))

    @staticmethod
    def get_datasets(path, transform=None, target_transform=None):
        if transform:
            assert isinstance(transform, list)

        transform_list = []
        if transform:
            transform_list.extend(transform)

        transform_list.append(transforms.ToTensor())
        train_dataset = datasets.MNIST(path, train=True, download=True,
                                       transform=transforms.Compose(transform_list),
                                       target_transform=target_transform)
        test_dataset = datasets.MNIST(path, train=False,
                                      transform=transforms.Compose(transform_list),
                                      target_transform=target_transform)
        return train_dataset, test_dataset
