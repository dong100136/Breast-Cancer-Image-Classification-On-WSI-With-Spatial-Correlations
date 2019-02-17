import pandas as pd
import numpy as np
import cv2
import os
import openslide
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor
pool = ThreadPoolExecutor(max_workers=20)

# config
base_data_path = "./data/origin"
img_size = 200
##


def save_img(csv_path, save_path):
    os.system("rm -rf "+save_path)
    os.mkdir(save_path)
    svs_imgs_name = ['A01', 'A02', 'A03', 'A04',
                     'A05', 'A06', 'A07', 'A08', 'A09', 'A10']

    svs_imgs = {img_name: openslide.OpenSlide(
        os.path.join(base_data_path, img_name, img_name+'.svs'))
        for img_name in svs_imgs_name}

    header = ['name', 'x', 'y', 'c1', 'c2',
              'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    d = pd.read_csv(csv_path, header=None, sep=',')
    d.columns = header

    f = open(os.path.join(save_path, 'list.txt'), 'w')
    ts = []
    for i, row in d.iterrows():
        path = os.path.join(save_path, str(row['c5']), str(i)+'.jpg')
        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))

        f.write(path+','+str(row['c5'])+"\n")

        def save_img(path, row):
            img = svs_imgs[row['name']]
            x = row['x']-img_size//2
            y = row['y']-img_size//2
            patch = img.read_region((y, x), 0, (img_size, img_size))
            patch.convert("RGB").save(path)

        ts.append(pool.submit(save_img, path, row))

    for t in tqdm(ts):
        t.done()

    f.close()


save_img("./data/origin/train_point_"+str(img_size) + ".csv",
         "./data/train_data_"+str(img_size)+"")
save_img("./data/origin/valid_point_"+str(img_size) + ".csv",
         "./data/valid_data_"+str(img_size)+"")
save_img("./data/origin/test_point_"+str(img_size) + ".csv",
         "./data/test_data_"+str(img_size)+"")
