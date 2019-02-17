import tensorflow as tf
import pandas as pd
import random
import numpy as np
import openslide
import os
import keras
from keras.datasets import mnist
from keras.layers import *
from keras.models import Sequential, Model, load_model
from keras.callbacks import *
from keras.utils import multi_gpu_model
from concurrent.futures import ThreadPoolExecutor
import cv2
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2

import matplotlib.pyplot as plt
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# config
base_data_path = "./data/origin"
image_num = 0
data = [[] for _ in range(4)]
batch_size = 64
patch_size = 200
model_name = "inceptionV3_200_hist"
train_data_path = "./data/train_data_"+str(patch_size)+"_hist"
valid_data_path = "./data/valid_data_"+str(patch_size)+"_hist"
test_data_path = "./data/test_data_"+str(patch_size)+"_hist"
checkpoint_path = './code/resnet_pretrain_ckpt_' + \
    str(patch_size)+'_%s.h5' % model_name
##

# model = ResNet50(weights='imagenet', include_top=False)
# model = VGG16(weights='imagenet', include_top=False)
# model = InceptionV3(weights='imagenet',include_top=False)
model = InceptionResNetV2(weights='imagenet', include_top=False)
x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512)(x)
x = ReLU()(x)
x = Dense(4)(x)
x = Softmax()(x)
model = Model(input=model.input, output=x)
# model = multi_gpu_model(model, gpus=2)
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=1e-3), metrics=['accuracy'])
model.summary()

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=lambda x: x/255.0)
valid_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=lambda x: x/255.0)

train_generator = train_datagen.flow_from_directory(
    train_data_path, target_size=(patch_size, patch_size), batch_size=batch_size, class_mode='categorical')
valid_generator = valid_datagen.flow_from_directory(
    valid_data_path, target_size=(patch_size, patch_size), batch_size=batch_size, class_mode='categorical')


checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
                             verbose=1, save_best_only=True)

os.system("rm -rf ./logs/"+model_name)
tensorboard = TensorBoard(log_dir='./logs/'+model_name)

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2,
                               mode='min', min_lr=1e-16, verbose=1)

early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, verbose=1, min_delta=0.01)
callbacks = [checkpoint, lr_reducer, tensorboard, early_stopping]

# model = load_model("./code/resnet_pretrain_ckpt.h5")
model.fit_generator(train_generator,
                    steps_per_epoch=train_generator.__len__(),
                    epochs=200,
                    validation_data=valid_generator,
                    validation_steps=20,
                    use_multiprocessing=True,
                    callbacks=callbacks,
                    verbose=1,
                    workers=10)

test_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=lambda x: x/255.0)
test_generator = valid_datagen.flow_from_directory(
    test_data_path, target_size=(patch_size, patch_size), batch_size=batch_size, class_mode='categorical')
valid_datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=lambda x: x/255.0)
valid_generator = valid_datagen.flow_from_directory(
    valid_data_path, target_size=(patch_size, patch_size), batch_size=batch_size, class_mode='categorical')
print(model.evaluate_generator(train_generator, steps=train_generator.__len__()))
print(model.evaluate_generator(valid_generator, steps=valid_generator.__len__()))
print(model.evaluate_generator(test_generator, steps=test_generator.__len__()))
