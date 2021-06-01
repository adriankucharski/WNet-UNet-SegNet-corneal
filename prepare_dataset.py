
from glob import glob
import numpy as np
from others import imstand, SysPath, write_hdf5, save_pickle_dataset, mask_reshape, shuffle_dataset
from skimage import io
from random import randint
import configparser
import re
from pathlib import Path


def extract_rand_field_list(orglist:list, patch_per_im=200, patch_width=32, patch_height=32, gt='gt', field='field', org='org'):
    gtlist = [fn.replace(org, gt) for fn in orglist]
    roilist = [fn.replace(org, field) for fn in orglist]

    npatch = patch_per_im * len(orglist)
    print('Files: ', len(orglist), ' Patches: ', npatch)
    # Create space for data
    X = np.empty((npatch, patch_height, patch_width), np.float16)
    Y = np.empty((npatch, patch_height, patch_width), np.float16)

    # Extract data from images
    patch = 0
    hh = int(patch_height / 2)
    ww = int(patch_width  / 2)
    for (fno, fng, fnr) in zip(orglist, gtlist, roilist):
        # Load images as gray and normalize them
        org = io.imread(fno, as_gray=True)
        org = org/np.max(org)
        org = imstand(org)

        gt = io.imread(fng, as_gray=True)
        gt = gt/127
        roi = io.imread(fnr, as_gray=True)
        roi = roi/np.max(roi)

        height, width = gt.shape
        k = 0
        while k < patch_per_im:
            x_center = randint(ww, width - ww)
            y_center = randint(hh, height-hh)
            if roi[y_center, x_center] == 0:
                continue

            X[patch, :, :] = (org[y_center -
                                  hh:y_center+hh, x_center-ww:x_center+ww])
            Y[patch, :, :] = gt[y_center -
                                hh:y_center+hh, x_center-ww:x_center+ww]
            patch += 1
            k += 1
    print('Total patches extracted: ', patch, '/', npatch)

    X = np.expand_dims(X, axis=-1)
    Y = np.expand_dims(Y, axis=-1)
    X, Y = shuffle_dataset(X, Y)

    Y = np.array(Y, np.uint8)
    Y = mask_reshape(Y)
    return X, Y


def extract_rand_field(org_path, gt_path, roi_path, patch_per_im=200, patch_width=32, patch_height=32):
    # Get trainning image list
    orglist = [fn for fn in glob(org_path+'/*')]
    gtlist = [fn.replace(org_path, gt_path) for fn in orglist]
    roilist = [fn.replace(org_path, roi_path) for fn in orglist]

    npatch = patch_per_im * len(orglist)
    print('Files: ', len(orglist), ' Patches: ', npatch)
    # Create space for data
    X = np.empty((npatch, patch_height, patch_width), np.float16)
    Y = np.empty((npatch, patch_height, patch_width), np.float16)

    # Extract data from images
    patch = 0
    hh = int(patch_height/2)
    ww = int(patch_width / 2)
    for (fno, fng, fnr) in zip(orglist, gtlist, roilist):
        # Load images as gray and normalize them
        org = io.imread(fno, as_gray=True)
        org = org/np.max(org)
        org = imstand(org)

        gt = io.imread(fng, as_gray=True)
        gt = gt/127
        roi = io.imread(fnr, as_gray=True)
        roi = roi/np.max(roi)

        height, width = gt.shape
        k = 0
        while k < patch_per_im:
            x_center = randint(ww, width - ww)
            y_center = randint(hh, height-hh)
            if roi[y_center, x_center] == 0:
                continue

            X[patch, :, :] = (org[y_center -
                                  hh:y_center+hh, x_center-ww:x_center+ww])
            Y[patch, :, :] = gt[y_center -
                                hh:y_center+hh, x_center-ww:x_center+ww]
            patch += 1
            k += 1
    print('Total patches extracted: ', patch, '/', npatch)
    return X, Y


def get_fold_dataset_dict(folds=5, path='./Data/Full/org'):
    Alizarine = []
    Gavet = []
    Hard = []
    Rotterdam = []
    Full = [Alizarine, Gavet, Hard, Rotterdam]
    for filePath in sorted(glob(path + '/*.png'), key=lambda x: float(re.findall("(\d+)", x)[0])):
        name = Path(filePath).parts[-1]
        if 'A' in name:
            Alizarine.append(filePath)
        if 'G' in name:
            Gavet.append(filePath)
        if 'H' in name:
            Hard.append(filePath)
        if 'R' in name:
            Rotterdam.append(filePath)
    Folds = dict()
    for k in range(folds):
        Folds[f'Fold_{k}'] = {'Test': [], 'Train': []}

    for dataset in Full:
        for k in range(folds):
            testIndex = k
            for i in range(0, len(dataset)):
                if i == testIndex:
                    Folds[f'Fold_{k}']['Test'].append(dataset[i])
                    testIndex += folds
                else:
                    Folds[f'Fold_{k}']['Train'].append(dataset[i])
    return Folds

#--------------------------------------------------------------------------------------#
if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    folds_number = int(config['Data']['folds_number'])
    patch_per_image = int(config['Data']['patch_per_image'])

    patch_size_x = int(config['Data']['patch_size_x'])
    patch_size_y = int(config['Data']['patch_size_y'])

    path_dataset = config['Data']['path_dataset']
    gt = config['Data']['path_image_gt']
    org = config['Data']['path_image_org']
    field = config['Data']['path_image_field']


    Folds = get_fold_dataset_dict(folds_number, str(Path(path_dataset) / org))

    for fold in Folds:
        X_train, Y_train = extract_rand_field_list(Folds[fold]['Train'], patch_per_image, patch_size_x, patch_size_y, gt, field, org)
        save_pickle_dataset((X_train, Y_train), f'{path_dataset}{fold}.pickle')
