import math
import torch
import numpy as np
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler


class ClassSampler(Sampler):
    """Sampler that restricts data loading to a single class of the dataset.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        class_number: The class index to filter out.
                      This can be a list as well to handle
                      multiple classes.
    """
    def __init__(self, target_class, reverse, class_number, shuffle=True):
        assert class_number is not None
        self.class_number = class_number
        self.shuffle = shuffle
        self.target_class = target_class
        self.reverse = reverse

    def __call__(self, dataset, class_number=None):
        # return dataset
        ''' helps to recompute indices '''
        if class_number is None:
            class_number = 1 - self.class_number if self.reverse else self.class_number

        # if we receive a list, then iterate over this sequentially
        self.indices = [
            i for i, (_, target) in enumerate(dataset)
            if target[self.target_class] == class_number
        ]
        self.num_samples = len(self.indices)
        # DEBUG print:
        # print("#indices for {} = {} | dataset = {}".format(self.class_number,
        #                                                    len(self.indices),
        #                                                    len(self.dataset)))

        # set the current dataset as a subset
        self.dataset = Subset(dataset, self.indices)
        return self.dataset

    @staticmethod
    def _calc_indices(dataset, class_number):
        indices = [
            i for i, (_, target) in enumerate(dataset)
            if target == class_number
        ]
        return indices, len(indices)

    def __iter__(self):
        assert hasattr(
            self, 'indices'), "need to run __call__() on ClassSampler first"
        if self.shuffle:
            return (self.indices[i] for i in torch.randperm(len(self.indices)))

        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return self.num_samples
