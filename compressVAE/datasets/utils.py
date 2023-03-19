# import cv2
import torch
import numpy as np

from copy import deepcopy
from PIL import Image
from collections import namedtuple
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from datasets.class_sampler import ClassSampler

# simple namedtuple loader
GenericLoader = namedtuple('GenericLoader', 'img_shp output_size train_loader test_loader')


def resize_lambda(img, size=(64, 64)):
    if not isinstance(img, (np.float32, np.float64)):
        img = np.asarray(img)

    if not isinstance(size, tuple):
        size = tuple(size)

    return cv2.resize(img, size)


def permute_lambda(img, pixel_permutation):
    if not isinstance(img, (np.float32, np.float64)):
        img = np.asarray(img)

    img_orig_shape = img.shape
    return Image.fromarray(
        img.reshape(-1, 1)[pixel_permutation].reshape(img_orig_shape)
    )


def bw_2_rgb_lambda(img):
    if img.mode == "RGB":
        return img

    return img.convert(mode="RGB")

def binarize(img, block_size=21):
    if not isinstance(img, (np.float32, np.float64)):
        img = np.asarray(img)

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, 0)
    return np.expand_dims(img, -1) if len(img.shape) < 3 else img


def find_max_label(loader):
    max_label = 0
    for data, lbls in loader:
        max_seen_lbl = max(lbls)
        if max_seen_lbl > max_label:
            max_label = max_seen_lbl

    return max_label


def label_offset_merger(loaders, batch_size, use_cuda=False):
    ''' iterate over all the loaders and:
           1. finds the max labels
           2. increments loader2 with +loader1_max_label
           3. build a new loader with all the data [uses simple_merger]'''
    # step 1
    max_labels_train = [find_max_label(loader.train_loader) for loader in loaders]
    max_labels_test = [find_max_label(loader.test_loader) for loader in loaders]
    max_labels = np.maximum(max_labels_test, max_labels_train) + 1
    for j in range(1, len(max_labels)):
        max_labels[j] += max_labels[j - 1]

    print('determined offset max_labels: ', max_labels)
    max_labels = torch.from_numpy(max_labels.astype(np.int32))

    # step 2
    def _extract_and_increment(loader, idx):
            data_container, lbl_container = [], []
            for data, labels in loader: # extract all the data
                data_container.append(data)
                lbl_container.append(labels)

            # handle data concat
            if isinstance(data_container[0], torch.Tensor):
                data_container = torch.cat(data_container, 0)
            elif isinstance(data_container[0], np.array):
                data_container = torch.from_numpy(np.vstack(data_container))
            else:
                raise Exception("unknown data type")

            # handle label concat
            if isinstance(lbl_container[0], torch.Tensor):
                lbl_container = torch.cat(lbl_container, 0)
            elif isinstance(lbl_container[0], np.array):
                lbl_container = torch.from_numpy(np.vstack(lbl_container))
            else:
                raise Exception("unknown label type")

            # do the actual incrementing
            lbl_container += max_labels[idx - 1]
            dataset = torch.utils.data.TensorDataset(data_container, lbl_container)
            kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
            return torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=loader.batch_size,
                drop_last=True,
                shuffle=True,
                **kwargs
            )

    # recreate the fucking loaders
    for i in range(1, len(max_labels)):
        loaders[i].train_loader = _extract_and_increment(loaders[i].train_loader, i)
        loaders[i].test_loader = _extract_and_increment(loaders[i].test_loader, i)
        loaders[i].output_size = max_labels[i].cpu().item()

    # step3: finally merge them with simpleMerger
    return simple_merger(loaders, batch_size, use_cuda)


def simple_merger(loaders, batch_size, use_cuda=False):
    print("""\nWARN [simplemerger]: no process in place for handling different classes,
ignore this if you called label_offset_merger\n""")
    train_dataset = loaders[0].train_loader.dataset
    test_dataset = loaders[0].test_loader.dataset
    img_shp = loaders[0].img_shp
    output_size = loaders[-1].output_size

    for loader in loaders[1:]:
        train_dataset += loader.train_loader.dataset
        test_dataset += loader.test_loader.dataset
        assert img_shp == loader.img_shp
        #assert output_size == loader.output_size

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        **kwargs
    )

    return GenericLoader(img_shp,
                         output_size,
                         train_loader,
                         test_loader)


def create_loader(dataset, sampler, batch_size, shuffle, **kwargs):
    if isinstance(sampler, ClassSampler):
        # our sampler is hacky; just filters dataset
        # and nulls itself out for GC
        dataset = sampler(dataset)
        sampler = None

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=shuffle,
        sampler=sampler,
        **kwargs)

def sequential_test_set_merger(loaders):
    test_dataset = [loaders[0].test_loader.dataset]
    for loader in loaders[1:]:
        current_clone = deepcopy(loader.test_loader.dataset)
        for td in test_dataset:
            loader.test_loader.dataset += td

        # re-create the test loader
        # in order to get correct samples
        loader.test_loader \
            = create_loader(loader.test_loader.dataset,
                            None, #loader.test_loader.sampler,
                            loader.batch_size,
                            shuffle=True)
        test_dataset.append(current_clone)

    return loaders
