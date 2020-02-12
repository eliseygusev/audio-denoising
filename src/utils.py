import os

import numpy as np
import torch


def files_in_subdirectory(path_to_dir):
    files = []
    for subdir_path, _, subdir_files in os.walk(path_to_dir):
        files += [os.path.join(subdir_path, x) for x in subdir_files]
    return files

def preprocess(sample, size):
    # preprocess sample to have given size.
    len_sample = len(sample)
    sample = sample.T
    # pad with 0 to make sample length divisible by size[1]
    sample = np.pad(sample, ((0, 0), (0, size[1] - len_sample % size[1])))
    sample = sample.reshape(-1, size[0], size[1])
    sample = np.expand_dims(sample, axis=1) # add empty channel dimension
    return torch.Tensor(sample)

def postprocess(mask, size):
    # reshape mask to original size
    return mask.reshape(-1, size[1])[:size[0], :]
