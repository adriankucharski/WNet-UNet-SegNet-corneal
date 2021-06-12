import configparser
import pickle

import matplotlib.pyplot as plt

from others import SysPath


def create_plot(history, save_plot = True):
    try:
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        tr_m = ('%.4f'%max(history['accuracy']))
        vl_m = ('%.4f'%max(history['val_accuracy']))
        plt.legend([ 'Train max = ' + tr_m, 'Test max = ' +vl_m ], loc='lower right')
        plt.ylim(bottom=0.8, top=1.0)
        if save_plot == True:
            plt.savefig(SysPath('Trained_model/accuracy.png'))
        plt.show()
    except Exception as e:
        print(e)
    try:
        plt.clf()
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        tr_m = ('%.4f'%min(history['loss']))
        vl_m = ('%.4f'%min(history['val_loss']))
        plt.legend([ 'Train min = ' + (tr_m), 'Test min = '+ (vl_m)], loc='upper left')
        plt.ylim(bottom=0.10, top=0.3)
        if save_plot == True:
            plt.savefig('Trained_model/loss.png')
        plt.show()
    except Exception as e:
        print(e)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    history_path = f'./Trained_model/UNet/history_Fold_0.pickle'

    history = pickle.load(open(SysPath(history_path), 'rb'))
    create_plot(history)
