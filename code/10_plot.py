import numpy as np
import pandas as pd
from PIL import Image
import math
import openslide
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import cv2

# config
model_name = "inceptionResNetV2"
base_path = "./data/rs_hist_%s" % model_name
model_path = os.path.join(
    base_path, "resnet_pretrain_ckpt_200_%s.h5" % model_name)
csv_data_path = os.path.join(base_path, "predict_map.csv")
svs_img_names = ['A01', 'A02', 'A03', 'A04',
                 'A05', 'A06', 'A07', 'A08',
                 "A09", "A10"]
svs_base_path = "./data/origin/%s/%s.svs"
svs_imgs = {svs_name: openslide.OpenSlide(svs_base_path % (
    svs_name, svs_name)) for svs_name in svs_img_names}

patch_size = 200

svs_size = {svs_name: [svs_imgs[svs_name].level_dimensions[0][1]//patch_size*10,
                       svs_imgs[svs_name].level_dimensions[0][0]//patch_size*10]
            for svs_name in svs_img_names
            }

gt_imgs = {
    svs_name: np.zeros([svs_size[svs_name][0], svs_size[svs_name][1], 3])
    for svs_name in svs_img_names
}

cnn_imgs = {
    svs_name: np.zeros([svs_size[svs_name][0], svs_size[svs_name][1], 3])
    for svs_name in svs_img_names
}

color_maps = {
    0: np.array([255, 255, 255]),
    1: np.array([255, 0, 0]),
    2: np.array([0, 255, 0]),
    3: np.array([0, 0, 255])
}

# end config

with open(csv_data_path, 'r') as f:
    for line in tqdm(f.readlines()):
        img_name, x, y, label, preds1, preds2, preds3, preds4 = line.strip().split(',')

        x = (int(x)-patch_size//2)//patch_size*10
        y = (int(y)-patch_size//2)//patch_size*10
        label = int(label)
        preds = list(map(float, [preds1, preds2, preds3, preds4]))
        preds_label = np.argmax(preds, axis=-1)

        gt_imgs[img_name][x:x+10, y:y+10] = color_maps[label]
        cnn_imgs[img_name][x:x+10, y:y+10] = color_maps[preds_label]

for svs_name in svs_img_names:
    gt = gt_imgs[svs_name][:, :, -1::-1]
    cv2.imwrite(os.path.join(base_path, svs_name+"_gt.png"), gt)

    cnn = cnn_imgs[svs_name][:, :, -1::-1]
    cv2.imwrite(os.path.join(base_path, svs_name+"_cnn.png"), cnn)
