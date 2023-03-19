from torchvision import datasets, transforms

from datasets.utils import create_loader


class CIFAR10Loader(object):
    def __init__(self,
                 path,
                 train_batch_size,
                 test_batch_size,
                 train_sampler=None,
                 test_sampler=None,
                 transform=None,
                 target_transform=None,
                 use_cuda=1,
                 **kwargs):
        # first get the datasets
        train_dataset, test_dataset = self.get_datasets(
            path, transform, target_transform)

        # build the loaders
        kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
        self.train_loader = create_loader(
            train_dataset,
            train_sampler,
            train_batch_size,
            shuffle=True if train_sampler is None else False,
            **kwargs)

        self.test_loader = create_loader(test_dataset,
                                         test_sampler,
                                         test_batch_size,
                                         shuffle=False,
                                         **kwargs)

        self.output_size = 10
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.img_shp = [3, 32, 32]

    @staticmethod
    def get_datasets(path, transform=None, target_transform=None):
        if transform:
            assert isinstance(transform, list)

        transform_list = []
        if transform:
            transform_list.extend(transform)

        transform_list.append(transforms.ToTensor())
        train_dataset = datasets.CIFAR10(
            path,
            train=True,
            download=True,
            transform=transforms.Compose(transform_list),
            target_transform=target_transform)
        test_dataset = datasets.CIFAR10(
            path,
            train=False,
            transform=transforms.Compose(transform_list),
            target_transform=target_transform)
        return train_dataset, test_dataset
