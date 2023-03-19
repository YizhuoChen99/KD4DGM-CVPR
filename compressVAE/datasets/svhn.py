from __future__ import print_function
from torchvision import datasets, transforms

from datasets.utils import create_loader


class SVHNCenteredLoader(object):
    def __init__(self, path, batch_size, train_sampler=None, test_sampler=None,
                 transform=None, target_transform=None, use_cuda=1, **kwargs):
        # first grab the datasets
        train_dataset, test_dataset = self.get_datasets(
            #path, target_transform=None #transforms.Lambda(lambda lbl: lbl - 1)
            path, transform, target_transform
        )

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
        self.img_shp = [3, 32, 32]

    @staticmethod
    def get_datasets(path, transform=None, target_transform=None):
        if transform:
            assert isinstance(transform, list)

        transform_list = []
        if transform:
            transform_list.extend(transform)

        # target_transform_list = [transforms.Lambda(lambda lbl: lbl - 1)]
        # if target_transform:
        #     target_transform_list.append(target_transform)

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        # normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5),
        #                                  std=(0.5, 0.5, 0.5))
        # transform_list.append(normalize)
        transform_list.append(transforms.ToTensor())

        train_dataset = datasets.SVHN(path, split='train', download=True,
                                      transform=transforms.Compose(transform_list),
                                      target_transform=target_transform)
                                      #target_transform=transforms.Compose(target_transform_list))
        test_dataset = datasets.SVHN(path, split='test', download=True,
                                     transform=transforms.Compose(transform_list),
                                     target_transform=target_transform)
                                     #target_transform=transforms.Compose(target_transform_list))
        return train_dataset, test_dataset
