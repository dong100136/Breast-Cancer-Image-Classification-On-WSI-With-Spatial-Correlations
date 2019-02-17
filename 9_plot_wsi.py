import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import math
import openslide
from keras.models import load_model
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import keras.backend as K
import keras
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# config
base_path = "./data/rs_hist_inceptionV3"
model_path = os.path.join(
    base_path, "resnet_pretrain_ckpt_200_inceptionV3.h5")
csv_data_path = "./data/origin/origin_200_hist.csv"
svs_img_names = ['A01', 'A02', 'A03', 'A04',
                 'A05', 'A06', 'A07', 'A08',
                 "A09", "A10"]
svs_base_path = "./data/origin/%s/%s.svs"
svs_imgs = {svs_name: openslide.OpenSlide(svs_base_path % (
    svs_name, svs_name)) for svs_name in svs_img_names}
patch_size = 200
# end config


class Seq(keras.utils.Sequence):
    def __init__(self, csv_data_path, batch_size):
        self._csv_data_path = csv_data_path
        self._batch_size = batch_size
        self._preprocessing()

    def _preprocessing(self):
        self._csv_data = []
        with open(self._csv_data_path, 'r') as f:
            for line in f:
                img_name, x, y, c1, c2, c3, c4, c5, c6, c7, c8, c9 = line.strip().split(
                    ',')
                self._csv_data.append([img_name, int(x), int(y), int(c5)])

    def __len__(self):
        return math.ceil(len(self._csv_data)/self._batch_size)

    def __getitem__(self, idx):
        beiginId = idx*self._batch_size
        endId = beiginId+self._batch_size

        rows = self._csv_data[beiginId:endId]
        imgs = []
        labels = np.zeros((len(rows), 4))
        for i, (img_name, x, y, c5) in enumerate(rows):
            x = x-patch_size//2
            y = y-patch_size//2

            img = svs_imgs[img_name].read_region(
                (y, x), 0, (patch_size, patch_size))
            img = img.convert("RGB")
            img = np.asarray(img)
            img = img/255
            imgs.append(img)
            labels[i, c5] = 1
        return np.array(imgs), labels, rows


model = load_model(model_path)

seq = Seq(csv_data_path, batch_size=100)

preds = model.predict_generator(
    seq, steps=seq.__len__(), use_multiprocessing=True, workers=10, verbose=1)


rows = seq._csv_data
print(accuracy_score([x[3]for x in rows], np.argmax(preds, axis=-1)))
with open(os.path.join(base_path, "predict_map.csv"), 'w') as f:
    for (img_names, x, y, c5), pred in tqdm(zip(rows, preds)):
        l = "%s,%d,%d,%d,%s\n" % (
            img_names, x, y, c5, ','.join(map(str, pred)))
        f.write(l)
