import openslide
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from keras.models import load_model
from concurrent.futures import ThreadPoolExecutor
import os
import tensorflow as tf
import keras.backend as K
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

pool = ThreadPoolExecutor(max_workers=10)

# config
img_size = 200
svs_imgs_name = ['A01', 'A02', 'A03', 'A04',
                 'A05', 'A06', 'A07', 'A08', 'A09', 'A10']
csv_path = "./data/origin/origin_"+str(img_size)+".csv"
svs_path = "./data/origin/%s/%s.svs"
model_path = "./data/rs/resnet_pretrain_ckpt_200.h5"
# end config

model = load_model(model_path)


def plot_img(img_name):
    svs_img = openslide.OpenSlide(svs_path % (img_name, img_name))
    svs_size = [svs_img.level_dimensions[0][1], svs_img.level_dimensions[0][0]]

    # 读取csv
    csv_data = []
    with open(csv_path, 'r') as f:
        for line in f:
            row = line.split(',')
            if row[0] == img_name:
                csv_data.append(line.split(','))

    rs = np.zeros((svs_size[0]//img_size*10, svs_size[1]//img_size*10, 3))
    gt = np.zeros((svs_size[0]//img_size*10, svs_size[1]//img_size*10, 3))

    ts = []
    for row in csv_data:
        temp_img_name, xx, yy, c1, c2, c3, c4, c5, c6, c7, c8, c9 = row
        xx = int(xx)
        yy = int(yy)
        c5 = int(c5)

        def get_clazz(xx, yy, c5):
            mini_img = svs_img.read_region(
                (yy, xx), 0, (img_size, img_size))
            mini_img = mini_img.convert("RGB")
            mini_img = np.asarray(mini_img)
            mini_img = np.expand_dims(mini_img, axis=0)/255.0

            return mini_img, xx//img_size, yy//img_size, c5

        ts.append(pool.submit(get_clazz, xx, yy, c5))

    for t in tqdm(ts):
        t.done()
        mini_img, x, y, c5 = t.result()
        preds = model.predict_on_batch(mini_img)
        preds = np.argmax(preds, axis=-1)[0]
        # print(temp_img_name)
        # print(np.shape(svs_rs[temp_img_name]))
        if preds == 0:
            rs[10*x:10*x+10, 10*y:10*y+10] = [255, 255, 255]
        elif preds == 1:
            rs[10*x:10*x+10, 10*y:10*y+10] = [255, 0, 0]
        elif preds == 2:
            rs[10*x:10*x+10, 10*y:10*y+10] = [0, 255, 0]
        elif preds == 3:
            rs[10*x:10*x+10, 10*y:10*y+10] = [0, 0, 255]

        if c5 == 0:
            gt[10*x:10*x+10, 10*y:10*y+10] = [255, 255, 255]
        elif c5 == 1:
            gt[10*x:10*x+10, 10*y:10*y+10] = [255, 0, 0]
        elif c5 == 2:
            gt[10*x:10*x+10, 10*y:10*y+10] = [0, 255, 0]
        elif c5 == 3:
            gt[10*x:10*x+10, 10*y:10*y+10] = [0, 0, 255]

    rs = rs[:, :, -1::-1]
    cv2.imwrite("./data/rs/%s.png" % img_name, rs)

    gt = gt[:, :, -1::-1]
    cv2.imwrite("./data/rs/%s_gt.png" % img_name, gt)


for img_name in svs_imgs_name:
    plot_img(img_name)
