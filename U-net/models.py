import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import Add, Multiply, concatenate, Lambda, Input
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K


def UpSampling2DBilinear(size):
    return Lambda(lambda x: tf.image.resize_bilinear(x, size, align_corners=True))

def unet(input_size=(224, 224, 3), bn = True):
    inputs = Input(input_size)
    # Block 1
    conv1 = Conv2D(64, 3, 
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1) if bn else conv1
    conv1 = Conv2D(64, 3, 
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1) if bn else conv1
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Block 2
    conv2 = Conv2D(128, 3, 
                   activation='relu', 
                   padding='same', 
                   kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2) if bn else conv2
    conv2 = Conv2D(128, 3, 
                   activation='relu', 
                   padding='same', 
                   kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2) if bn else conv2
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Block 3
    conv3 = Conv2D(256, 3, 
                   activation='relu', 
                   padding='same', 
                   kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3) if bn else conv3
    conv3 = Conv2D(256, 3, 
                   activation='relu', 
                   padding='same', 
                   kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3) if bn else conv3
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Block 4
    conv4 = Conv2D(512, 3, 
                   activation='relu', 
                   padding='same', 
                   kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4) if bn else conv4
    conv4 = Conv2D(512, 3, 
                   activation='relu', 
                   padding='same', 
                   kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4) if bn else conv4
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    # Bottleneck 
    conv5 = Conv2D(1024, 3, 
                   activation='relu', 
                   padding='same', 
                   kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5) if bn else conv5
    conv5 = Conv2D(1024, 3, 
                   activation='relu', 
                   padding='same', 
                   kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5) if bn else conv5
    drop5 = Dropout(0.5)(conv5)
    
    # Upsampling Block 4
    upsampling6 = Conv2D(512, 2, 
                         activation='relu', 
                         padding='same', 
                         kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, upsampling6], axis=3)
    conv6 = Conv2D(512, 3, 
                   activation='relu', 
                   padding='same', 
                   kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6) if bn else conv6
    conv6 = Conv2D(512, 3, 
                   activation='relu', 
                   padding='same', 
                   kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6) if bn else conv6
    
    # Upsampling Block 3
    upsampling7 = Conv2D(256, 2, 
                         activation='relu', 
                         padding='same', 
                         kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, upsampling7], axis=3)
    conv7 = Conv2D(256, 3, 
                   activation='relu', 
                   padding='same', 
                   kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7) if bn else conv7
    conv7 = Conv2D(256, 3, 
                   activation='relu', 
                   padding='same', 
                   kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7) if bn else conv7
    
    # Upsampling Block 2
    upsampling8 = Conv2D(128, 2, 
                         activation='relu', 
                         padding='same', 
                         kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, upsampling8], axis=3)
    conv8 = Conv2D(128, 3, 
                   activation='relu', 
                   padding='same', 
                   kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8) if bn else conv8
    conv8 = Conv2D(128, 3, 
                   activation='relu', 
                   padding='same', 
                   kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8) if bn else conv8
    
    # Upsampling Block 1
    upsampling9 = Conv2D(64, 2, 
                         activation='relu', 
                         padding='same', 
                         kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, upsampling9], axis=3)
    conv9 = Conv2D(64, 3, 
                   activation='relu', 
                   padding='same', 
                   kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9) if bn else conv9
    conv9 = Conv2D(64, 3, 
                   activation='relu', 
                   padding='same', 
                   kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9) if bn else conv9
    conv9 = Conv2D(2, 3, 
                   activation='relu', 
                   padding='same', 
                   kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9) if bn else conv9
    
    output = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=output)

    return model







