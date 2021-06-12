import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import configparser
from tensorflow.keras.callbacks import ModelCheckpoint
from others import SysPath, dump_history, load_pickle_dataset

import models
from importlib import reload

reload(models)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    folds_number = int(config['Data']['folds_number'])
    patch_size = int(config['Data']['patch_size_x']), int(config['Data']['patch_size_y'])

    model_names = config['Training']['network'].split(',')
    batch_size = int(config['Training']['batch_size'])
    N_epochs = int(config['Training']['N_epochs'])
    val_split = float(config['Training']['val_split'])

    models_func = [(name.strip(), getattr(models, name.strip())((*patch_size, 1))) for name in model_names]

    for model_name, model in models_func:
        for fold_id in range(folds_number):
            fold = f'Fold_{fold_id}'
            Path_dataset = f'{SysPath(config["Data"]["path_dataset"])}/{fold}.pickle'

            X, Y = load_pickle_dataset(Path_dataset)
            print(model_name, X.shape, Y.shape)

            history_path_save = f'./Trained_model/{model_name}/history_{fold}.pickle'
            model_path_save = f'./Trained_model/{model_name}/model_{fold}.json'
            model_best_weight_save = f'./Trained_model/{model_name}/model_weights_{fold}.h5'

            os.makedirs(f'./Trained_model/{model_name}/', exist_ok=True)
            open(model_path_save, 'w').write(model.to_json())
            checkpointer = ModelCheckpoint(model_best_weight_save, verbose=2, monitor='val_loss', mode='auto',
                                           save_best_only=True)

            history = model.fit(X, Y, epochs=N_epochs, batch_size=batch_size, verbose=1,
                                shuffle=True, validation_split=val_split, callbacks=[checkpointer])

            dump_history(history, history_path_save)
