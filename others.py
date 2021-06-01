import numpy as np
from numpy.random import RandomState
import pickle
import h5py
import pathlib
import os

#-------------------------------------------------------#
#----------------------SYSTEM---------------------------#
#-------------------------------------------------------#
def SysPath(path):
    path = pathlib.Path(path)
    if os.name == 'nt':
        return str(pathlib.PureWindowsPath(path))
    return str(pathlib.Path(path))

#-------------------------------------------------------#
#----------------------NETWORK--------------------------#
#-------------------------------------------------------#
def dump_history(history, filename = None):
    filename = filename if filename is not None else 'history.pickle'
    with open(filename, 'wb') as fp:
        pickle.dump(history.history, fp)

#-------------------------------------------------------#
#----------------------DATA-----------------------------#
#-------------------------------------------------------#
def shuffle_dataset(x,y, seed = 1234):
    assert len(x) == len(y)
    state = RandomState(seed=seed)
    index = state.permutation(x.shape[0])
    return x[index], y[index]

def mask_reshape(mask, label = (0, 1, 2)):
    length, width, height = mask.shape[0],mask.shape[1],mask.shape[2]
    mask = np.reshape(mask, (length, width, height))
    new_masks = np.zeros((length, width, height, len(label)), dtype='float16')
    for i in range(0, len(label)):
        new_masks[0:length, 0:width, 0:height, i] = (mask == label[i])
    return new_masks

def load_dataset(org_path, gt_path, patch_size=(32, 32), label = (0,1,2)):
    patch_width = patch_size[1]
    patch_height = patch_size[0]

    X_train = load_hdf5(org_path)
    Y_train = load_hdf5(gt_path)
    patch_num = X_train.shape[0]
    X_train, Y_train = shuffle_dataset(X_train, Y_train)

    X_train = X_train.reshape((patch_num, patch_height, patch_width, 1))
    Y_train = Y_train.reshape((patch_num, patch_height, patch_width, 1))
    Y_train = np.array(Y_train, np.uint8)
    Y_train = mask_reshape(Y_train, label)
    return X_train, Y_train

def load_hdf5(infile):
    with h5py.File(infile, "r") as f:
        return f["image"][()]

def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)

def imstand(im):
    im = (im - np.mean(im)) / np.std(im)
    return im

def save_pickle_dataset(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle_dataset(path):
    with open(path, 'rb') as f:
        dataset = pickle.load(f)
        return dataset

