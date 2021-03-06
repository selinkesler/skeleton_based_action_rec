import bisect

import torch
import torch.utils.data
import torchvision

# Original src: https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/sampler.py
from nobos_torch_lib.datasets.action_recognition_datasets.ehpi_dataset import EhpiDataset
from nobos_torch_lib.datasets.action_recognition_datasets.ehpi_dataset_carlos import EhpiDataset_Carlos
from torch.utils.data import ConcatDataset, Subset


class ImbalancedDatasetSampler_Carlos(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None, dataset_type=None):
        self.dataset_type = dataset_type
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        actual_type = type(dataset)
        if self.dataset_type is None:
            self.dataset_type = type(dataset)
        if self.dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif self.dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        elif self.dataset_type is EhpiDataset_Carlos and actual_type is ConcatDataset:
            dataset_idx = bisect.bisect_right(dataset.cumulative_sizes, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - dataset.cumulative_sizes[dataset_idx - 1]
            if type(dataset.datasets[dataset_idx]) is Subset:
                return dataset.datasets[dataset_idx].dataset.y[sample_idx][0]
            return dataset.datasets[dataset_idx].y[sample_idx][0]
        elif self.dataset_type is EhpiDataset_Carlos and actual_type is Subset:
            return dataset.dataset.y[idx][0]
        elif self.dataset_type is EhpiDataset_Carlos:
            return dataset.y[idx][0]
        else:
            print(self.dataset_type)
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
