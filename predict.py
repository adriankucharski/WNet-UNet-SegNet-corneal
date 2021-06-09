import configparser
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
from skimage import io
from glob import glob
from others import SysPath, imstand
from tensorflow.keras.models import model_from_json
from prepare_dataset import get_fold_dataset_dict


def build_img_from_patches(preds, img_h, img_w, stride_h, stride_w):
    patch_h, patch_w = preds.shape[1],  preds.shape[2]

    H = (img_h-patch_h)//stride_h+1
    W = (img_w-patch_w)//stride_w+1
    prob = np.zeros((img_h, img_w, 3))
    _sum = np.zeros((img_h, img_w, 3))

    k = 0
    for h in range(H):
        for w in range(W):
            prob[h*stride_h:(h*stride_h)+patch_h, w *
                 stride_w:(w*stride_w)+patch_w, :] += preds[k]
            _sum[h*stride_h:(h*stride_h)+patch_h, w *
                 stride_w:(w*stride_w)+patch_w, :] += 1
            k += 1
    final_avg = prob/_sum
    return final_avg
#--------------------------------------------------------------------------------------#

def pred_to_img(pred, patch_h, patch_w):
    pred_image = np.zeros((pred.shape[0], pred.shape[1], pred.shape[2], 3))
    pred_image[0:pred.shape[0], 0:pred.shape[1], 0:pred.shape[2], 0:pred.shape[3]
               ] = pred[0:pred.shape[0], 0:pred.shape[1], 0:pred.shape[2], 0:pred.shape[3]]
    return pred_image
#--------------------------------------------------------------------------------------#
def get_patches(img, patch_h, patch_w, stride_h, stride_w):
    h, w = img.shape[0], img.shape[1]
    assert ((h-patch_h) % stride_h == 0 and (w-patch_w) % stride_w == 0)

    H = (h-patch_h)//stride_h+1
    W = (w-patch_w)//stride_w+1

    patches = np.empty((W*H, patch_h, patch_w, 1))
    iter_tot = 0
    for h in range(H):
        for w in range(W):
            patches[iter_tot] = (img[h*stride_h:(h*stride_h)+patch_h,
                                     w*stride_w:(w*stride_w)+patch_w, :])
            iter_tot += 1
    return patches
#--------------------------------------------------------------------------------------#

def add_outline(img, patch_h, patch_w, stride_h, stride_w):
    img_h = img.shape[0]
    img_w = img.shape[1]
    leftover_h = (img_h-patch_h) % stride_h
    leftover_w = (img_w-patch_w) % stride_w

    if (leftover_h != 0):
        tmp = np.zeros((img_h+(stride_h-leftover_h), img_w, 1))
        tmp[0:img_h, 0:img_w, :] = img
        img = tmp
    if (leftover_w != 0):
        tmp = np.zeros((img.shape[0], img_w+(stride_w - leftover_w), 1))
        tmp[0:img.shape[0], 0:img_w] = img
        img = tmp
    return img
#--------------------------------------------------------------------------------------#

def predict_img(org_path, model, patch_height=32, patch_width=32, stride_height=4, stride_width=4, batch_size=32):
    org = io.imread(org_path, as_gray=True)
    org = np.asarray(imstand(org), dtype='float16')

    print(os.path.split(org_path)[-1],  org.shape)
    height, width = org.shape[0], org.shape[1]

    org = np.reshape(org, (height, width, 1))
    assert(org.shape == (height, width, 1))

    org = add_outline(org, patch_height, patch_width, stride_height, stride_width)
    new_height, new_width = org.shape[0], org.shape[1]
    org = np.reshape(org, (new_height, new_width, 1))

    assert(org.shape == (new_height, new_width, 1))

    patches = get_patches(org, patch_height, patch_width, stride_height, stride_width)
    predictions = model.predict(patches, batch_size=batch_size, verbose=1)

    pred_patches = pred_to_img(predictions, patch_height, patch_width)
    pred_img = build_img_from_patches(
        pred_patches, new_height, new_width, stride_height, stride_width)

    pred = pred_img[0:height, 0:width]
    return np.array(255*pred/np.max(pred), dtype=np.uint8)
#--------------------------------------------------------------------------------------#


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    patch_width = int(config['Data']['patch_size_x'])
    patch_height = int(config['Data']['patch_size_y'])

    batch_size = int(config['Predict']['batch_size'])
    stride_width = int(config['Predict']['stride_width'])
    stride_height = int(config['Predict']['stride_height'])

    models = config['Training']['network'].split(',')
    folds = config['Data']['fold'].split(',')
    folds_number = int(config['Data']['folds_number'])

    path_dataset = config['Data']['path_dataset']
    gt = config['Data']['path_image_gt']
    org = config['Data']['path_image_org']
    field = config['Data']['path_image_field']

    models_path = config['Network']['models_path']
    predict_save = config['Predict']['predict_save']
    predict_to = config['Predict']['to_predict']

    Folds = get_fold_dataset_dict(folds_number, path='./Training_data/org/')
    for fold in folds:
        fold = fold.strip()
        for net in models:
            net = net.strip()
            model_path = f'{models_path}{net}/model_{fold}.json'
            model_weights = f'{models_path}{net}/model_weights_{fold}.h5'
            model = model_from_json(open(model_path).read())
            model.load_weights(model_weights)

            for impath in Folds[fold]['Test']:
                imname = os.path.split(impath)[-1]
                path_to_save = f'{predict_save}{fold}/{net}/{imname}'
                os.makedirs(os.path.split(path_to_save)[0], exist_ok=True)

                pred = predict_img(impath, model, patch_height, patch_width, stride_height, stride_width, batch_size)
                io.imsave(path_to_save, pred)
