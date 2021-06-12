import tensorflow.keras.backend as K
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Dropout, Input, MaxPooling2D,
                                     UpSampling2D, concatenate)
from tensorflow.keras.models import Model


def dice_coef(y_true, y_pred):
    smooth = 1e-6
    intersection =  2 * K.sum(K.abs(y_true * y_pred), axis=-1) + smooth
    sums = (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)
    return intersection / sums
def dice_coef_multilabel(y_true, y_pred, numLabels=3):
    dice = 1
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index]) / numLabels
    return dice

def tversky_loss(y_true, y_pred, beta=0.5):
    # generalization of dice coefficient algorithm
    #   alpha corresponds to emphasis on False Positives
    #   beta corresponds to emphasis on False Negatives (our focus)
    #   if alpha = beta = 0.5, then same as dice
    #   if alpha = beta = 1.0, then same as IoU/Jaccard
    alpha = 1
    beta = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1-((intersection) / (intersection + alpha * (K.sum(y_pred_f*(1. - y_true_f))) + beta *  (K.sum((1-y_pred_f)*y_true_f))))
#--------------------------------------------------------------#
#---------------------------WNet-------------------------------#
#--------------------------------------------------------------#
def ConvWNetBlock(inputs, filters=32, kernel=(3, 3), dropout=0.2, ki = 'he_normal', act = 'relu'):
    conv = Conv2D(filters, kernel, padding="same", activation=act, kernel_initializer=ki)(inputs)
    conv = Conv2D(filters, kernel, padding="same", activation=act, kernel_initializer=ki)(conv)
    conv = Dropout(dropout)(conv)
    return conv

def WNet(shape=(32, 32, 1), weights=None):
    inputs = Input(shape)
    #
    conv1 = ConvWNetBlock(inputs, 32, (3, 3), 0.2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = ConvWNetBlock(pool1, 64, (3, 3), 0.2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = ConvWNetBlock(pool2, 128, (3, 3), 0.2)
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)
    #
    conv4 = ConvWNetBlock(up1, 64, (3, 3), 0.2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #
    conv5 = ConvWNetBlock(pool3, 128, (3, 3), 0.2)
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    #
    conv6 = ConvWNetBlock(up2, 64, (3, 3), 0.2)
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
    #
    conv7 = ConvWNetBlock(up3, 32, (3, 3), 0.2)

    outs = Conv2D(3, (1, 1), activation='softmax', padding='valid')(conv7)
    model = Model(inputs=inputs, outputs=outs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    if(weights):
        model.load_weights(weights)
    return model
#--------------------------------------------------------------#
#---------------------------UNet-------------------------------#
#--------------------------------------------------------------#
def ConvUNetBlock(inputs, filters=32, kernel=(3, 3), dropout=0.2, ki = 'he_normal', act = 'relu'):
    conv = Conv2D(filters, kernel, padding="same", activation=act, kernel_initializer=ki)(inputs)
    conv = Conv2D(filters, kernel, padding="same", activation=act, kernel_initializer=ki)(conv)
    conv = Dropout(dropout)(conv)
    return conv
def UNet(shape=(32, 32, 1), weights=None):
    inputs = Input(shape)
    #
    conv1 = ConvUNetBlock(inputs, 32, (3, 3), 0.20)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #
    conv2 = ConvUNetBlock(pool1, 64, (3, 3), 0.20)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = ConvUNetBlock(pool2, 128, (3, 3), 0.20)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #4x4
    conv4 = ConvUNetBlock(pool3, 128, (3, 3), 0.20)
    up1 = UpSampling2D(size=(2,2))(conv4)
    #
    con1 = concatenate([up1, conv3], axis=-1)
    conv5 = ConvUNetBlock(con1, 64, (3, 3), 0.20)
    up2 = UpSampling2D(size=(2,2))(conv5)
    #
    con2 = concatenate([up2, conv2], axis=-1)
    conv6 = ConvUNetBlock(con2, 32, (3, 3), 0.20)
    up3 = UpSampling2D(size=(2,2))(conv6)
    #
    con3 = concatenate([up3, conv1], axis=-1)
    conv7 = ConvUNetBlock(con3, 32, (3, 3), 0.20)

    outs = Conv2D(3, (1, 1), activation='softmax', padding='valid')(conv7)
    model = Model(inputs=inputs, outputs=outs)
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
    model.summary()
    if(weights):
        model.load_weights(weights)
    return model
#--------------------------------------------------------------#
#---------------------------SegNet-----------------------------#
#--------------------------------------------------------------#
def ConvSegNetBlock(inputs, filters, layers, kernel=(3,3),padd='same', act = 'relu', ki = 'he_normal', dr=0.05):
    conv = inputs
    for _ in range(layers):
        conv = Conv2D(filters, kernel, kernel_initializer=ki, padding=padd, activation=None)(conv)
        conv = BatchNormalization()(conv)
        conv = Activation(act)(conv)
        conv = Dropout(dr)(conv)
    return conv

def SegNet(shape=(32, 32, 1), weights=None):
    dr = 0.05
    inputs = Input(shape)
    
    conv1 = ConvSegNetBlock(inputs, 32, 2, dr=dr)
    pool1 = MaxPooling2D()(conv1)

    conv2 = ConvSegNetBlock(pool1, 32, 1, dr=dr)
    conv3 = ConvSegNetBlock(conv2, 64, 1, dr=dr)
    pool2 = MaxPooling2D()(conv3)

    conv4 = ConvSegNetBlock(pool2, 64, 1, dr=dr)
    conv5 = ConvSegNetBlock(conv4, 64, 2, dr=dr)
    pool3 = MaxPooling2D()(conv5)

    conv5_1 = ConvSegNetBlock(pool3, 128, 2, dr=dr)

    up1 = UpSampling2D()(conv5_1)
    conv6 = ConvSegNetBlock(up1, 128, 2, dr=dr)
    conv7 = ConvSegNetBlock(conv6, 64, 1, dr=dr)

    up2 = UpSampling2D()(conv7)
    conv8 = ConvSegNetBlock(up2, 64, 2, dr=dr)

    up3 = UpSampling2D()(conv8)
    conv9 = ConvSegNetBlock(up3, 32, 1, dr=dr)

    outs = Conv2D(3, (1, 1), activation='softmax', padding='valid')(conv9)
    model = Model(inputs=inputs, outputs=outs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    if(weights):
        model.load_weights(weights)

    return model
