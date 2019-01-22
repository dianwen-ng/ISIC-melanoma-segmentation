import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from model import *
from Datasets.ISIC2018 import *
import numpy as np
import os as os
import matplotlib.pyplot as plt

'''
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.
sess=tf.Session(config = config)
'''

np.random.seed(609)
num_folds = 5
k_fold = np.random.randint(0,5)

num_classes = 1
batch_size = 32
initial_epoch = 0
epochs = 25
init_lr = 1e-4
min_lr = 1e-7
patience = 1
loss = 'crossentropy'
metrics = ['jaccard_index',
           'pixelwise_sensitivity',
           'pixelwise_specificity']

horizontal_flip = True
vertical_flip = True
rotation_angle = 180
width_shift_range = 0.1
height_shift_range = 0.1

(x_train, y_train), (x_valid, y_valid), _ = load_training_data(output_size=224,
                                                               num_partitions=num_folds,
                                                               idx_partition=k_fold)

# Target should be of the shape: N x 224 x 224 x 1 
if len(y_train.shape) == 3:
    y_train = np.expand_dims(y_train, axis= -1)
    y_valid = np.expand_dims(y_valid, axis= -1)
    
# scaling mask 
y_train = (y_train > 127.5).astype(np.uint8)
y_valid = (y_valid > 127.5).astype(np.uint8)

model = unet(loss=loss, lr=init_lr ,metrics= metrics, num_classes=1)
#model.summary()

n_samples_train = x_train.shape[0]
n_samples_valid = x_valid.shape[0]
steps_per_epoch = n_samples_train//batch_size

data_gen_args = dict(horizontal_flip= horizontal_flip,
                     vertical_flip= vertical_flip,
                     rotation_range= rotation_angle,
                     width_shift_range= width_shift_range,
                     height_shift_range= height_shift_range)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 610
image_generator = image_datagen.flow(x=x_train, batch_size= batch_size, seed= seed)
mask_generator = mask_datagen.flow(x=y_train, batch_size= batch_size, seed= seed)

train_generator = zip(image_generator, mask_generator)
model.fit_generator(generator= train_generator,
                    steps_per_epoch= steps_per_epoch,
                    epochs = epochs,
                    initial_epoch = initial_epoch,
                    verbose= 1,
                    validation_data= (x_valid, y_valid),
                    workers = 16,
                    use_multiprocessing= False)