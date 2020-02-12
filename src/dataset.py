import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import files_in_subdirectory
from .config import SAMPLE_SIZE


class SpeechDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.filenames = files_in_subdirectory(root_dir)

    def __len__(self):
        return len(self.filenames)

    def _resize_and_pad(self, a, size=SAMPLE_SIZE):
        a = a[:size[0], :size[1]]
        result = np.zeros(size)
        result[:a.shape[0], :a.shape[1]] = a
        return result

    def __getitem__(self, idx):
        path_to_sample = self.filenames[idx]
        sample = np.expand_dims(self._pad(np.load(path_to_sample).T), axis=0)
        class_target = 0 if 'noisy' in path_to_sample else 1
        if 'noisy' in path_to_sample:
            clean_sample_path = path_to_sample.replace('noisy', 'clean')
            mask_target = np.expand_dims(self._resize_and_pad(
                np.load(clean_sample_path).T), axis=0)
        else:
            mask_target = sample
        return sample, mask_target, class_target
