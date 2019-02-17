import cv2
import openslide
import os
import numpy as np
import matplotlib.pylab as plt
import pyprind

# config
data_path = "./data/origin"
imgs = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10"]
# end config

for img in imgs:
    path = os.path.join(data_path, img)
    avaliable_path = os.path.join(path, "label.png")
    mask_path = os.path.join(path, img+".png")
    svs_path = os.path.join(path, img+".svs")

    img = openslide.OpenSlide(svs_path)
    img_size = img.level_dimensions[0]
    img_size = list(img_size)
    width = img_size[0]
    height = img_size[1]
    # img_size[0], img_size[1] = img_size[1], img_size[0]
    print(img_size)

    avaliable_mask = cv2.imread(avaliable_path)
    avaliable_mask = np.mean(avaliable_mask, axis=-1)
    avaliable_mask = cv2.resize(avaliable_mask, (width, height))
    avaliable_mask = avaliable_mask[:, :]

    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, (width, height))
    mask = mask[:, :, -1::-1]

    rs = np.zeros((height, width), np.int8)
    rs[avaliable_mask[:, :] > 0] = 5  # 正常细胞
    rs[mask[:, :, 0] > 0] = 1
    rs[mask[:, :, 1] > 0] = 2
    rs[mask[:, :, 2] > 0] = 3

    np.save(os.path.join(path, "mask.npy"), rs)

    rs = cv2.resize(rs/5*255, (width//100, height//100))
    cv2.imwrite(os.path.join(path, "mask.png"), rs)
    print(path)
