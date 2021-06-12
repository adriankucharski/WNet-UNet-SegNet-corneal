import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from skimage import io
from others import imstand
from tensorflow.keras.models import model_from_json
from prepare_dataset import get_fold_dataset_dict
from glob import glob
from pathlib import Path
from predict import predict_img
import configparser
if __name__ == '__main__':
    # Get value from config.ini
    config = configparser.ConfigParser()
    config.read('config.ini')
    patch_width = int(config['Data']['patch_size_x'])
    patch_height = int(config['Data']['patch_size_y'])
    batch_size = int(config['Predict']['batch_size'])
    stride_width = int(config['Predict']['stride_width'])
    stride_height = int(config['Predict']['stride_height'])

    # 'SegNet', 'WNet', 'UNet'
    network = 'WNet'

    # You can replace it by your own path
    path_to_images = './Training_data/org/'
    path_to_save = './predicted/'

    # Path to trained model, run training.py
    path_to_trained_model = f'./Trained_model/{network}/model_Fold_0.json'
    path_to_trained_model_weights = f'./Trained_model/{network}/model_weights_Fold_0.h5'

    # Load model
    model = model_from_json(open(path_to_trained_model).read())
    model.load_weights(path_to_trained_model_weights)

    for im_path in glob(str(Path(path_to_images) / '*')):
        im_name = os.path.split(im_path)[-1]
        save_name = f'{path_to_save}/{im_name}'
        os.makedirs(os.path.split(path_to_save)[0], exist_ok=True)

        prediction = predict_img(im_path, model, patch_height, patch_width, stride_height, stride_width, batch_size)
        io.imsave(save_name, prediction)
