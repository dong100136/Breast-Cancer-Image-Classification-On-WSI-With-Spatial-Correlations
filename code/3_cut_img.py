import cv2
import openslide
import numpy as np
import os
import random
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
pool = ThreadPoolExecutor(max_workers=12)

# config
base_data_path = "./data/origin"
imgs_name = ['A01', 'A02', 'A03', 'A04',
             'A05', 'A06', 'A07', 'A08', 'A09', 'A10']
img_size = 200
img_num_per_size = 3
# end of config

imgs_mask = [np.load(os.path.join(base_data_path, img_name, "mask.npy"))
             for img_name in imgs_name]

svs_imgs = [openslide.OpenSlide(os.path.join(
    base_data_path, img_name, img_name+".svs")) for img_name in imgs_name]

imgs_size = [(img.level_dimensions[0][1], img.level_dimensions[0][0])
             for img in svs_imgs]

out_data = [0 for _ in range(4)]


def get_class(svs, mask, center_point):
    x = center_point[0]-img_size*img_num_per_size//2
    y = center_point[1]-img_size*img_num_per_size//2

    rs = []
    for i in range(3):
        for j in range(3):
            xx = (x+img_size*i)
            yy = (y+img_size*j)

            crop = mask[xx:xx+img_size, yy:yy+img_size]
            all = np.shape(crop)[0]*np.shape(crop)[1]

            if len(crop[crop == 0]) > all*0.5:
                return None

            # 去除背景
            svs_crop = svs.read_region((yy, xx), 0, (img_size, img_size))
            svs_crop = svs_crop.convert('L')
            svs_crop = np.asarray(svs_crop)
            h, _ = np.histogram(svs_crop, bins=255, range=(0, 255))
            ratio = h[np.argmax(h)-1:np.argmax(h)+1].sum()/h.sum()
            if ratio > 0.5:
                return None

            c0 = len(crop[crop == 5])
            c1 = len(crop[crop == 1])
            c2 = len(crop[crop == 2])
            c3 = len(crop[crop == 3])

            if c3 > all*.5:
                rs.append(3)
            elif c2 > all*.5:
                rs.append(2)
            elif c1 > all*.5:
                rs.append(1)
            else:
                rs.append(0)

    return rs, center_point


# train and eval dataset
f = open("./data/origin/origin_"+str(img_size)+"_hist.csv", 'w')
for i, img_name in enumerate(imgs_name):
    svs = svs_imgs[i]
    safe_margin = img_size*img_num_per_size//2
    max_x, max_y = imgs_size[i]
    max_x, max_y = max_x-safe_margin-1, max_y-safe_margin-1

    num_x = (max_x-safe_margin)//img_size
    num_y = (max_y-safe_margin)//img_size

    img = imgs_mask[i]
    ts = []
    for x in range(num_x):
        for y in range(num_y):
            xx = safe_margin+img_size*x
            yy = safe_margin+img_size*y

            t = pool.submit(get_class, svs, img, (xx, yy))
            ts.append(t)

    for t in tqdm(ts):
        t.done()
        rs = t.result()
        if rs != None:
            clazz, (xx, yy) = rs
            if clazz != None:
                out_data[clazz[4]] += 1
                f.write(img_name+',')
                f.write(','.join(map(str, (xx, yy)))+',')
                f.write(','.join(map(str, clazz))+'\n')

f.flush()
f.close()
