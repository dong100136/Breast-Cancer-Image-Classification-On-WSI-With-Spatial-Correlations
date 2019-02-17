# 计算
import keras
import keras.backend as K
import tensorflow as tf
from keras.models import load_model
import os
import sklearn
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm
import random
import openslide
import pickle
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
pool = ThreadPoolExecutor(max_workers=12)

# config
patch_size = 200
patch_pre_side = 3
# end config
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

model = load_model("./code/resnet_pretrain_ckpt_200_inceptionResNetV2.h5")

imgs_name = ['A01', 'A02', 'A03', 'A04',
             'A05', 'A06', 'A07', 'A08',
             "A09", "A10"]
svs_imgs = {img_name: openslide.OpenSlide(
    os.path.join("./data/origin", img_name, img_name+".svs")) for img_name in imgs_name}


def parse_csv_data(csv_path):
    f = open(csv_path)
    csv_data = []
    for line in f:
        name, x, y, c1, c2, c3, c4, c5, c6, c7, c8, c9 = line.split(',')
        csv_data.append({'name': name, 'x': int(x), 'y': int(y), 'gt': np.array(
            list(map(int, [c1, c2, c3, c4, c5, c6, c7, c8, c9])))})
    return csv_data


def get_patches_with_center_point(svs_img, x, y):
    x = x-patch_pre_side*patch_size//2
    y = y-patch_pre_side*patch_size//2

    patches = []
    # img = svs_img.read_region((y, x),
    #                           0,
    #                           (patch_size*patch_pre_side, patch_size*patch_pre_side))
    # img = np.array(img)[:, :, :3]
    # for xx in range(0, patch_size*patch_pre_side, patch_size):
    #     for yy in range(0, patch_size*patch_pre_side, patch_size):
    #         patches.append(img[xx:xx+patch_size, yy:yy+patch_size])

    for i in range(3):
        for j in range(3):
            xx = x+patch_size*i
            yy = y+patch_size*j
            patch = svs_img.read_region((yy, xx), 0, (patch_size, patch_size))
            patch = np.asarray(patch.convert("RGB"))/255.0
#             print(np.shape(patch))
#             patch = np.array(patch)[:, :, :3]/255.0
            patches.append(patch)
    return np.array(patches)


def predict(patches):
    preds = model.predict(patches)
    return preds


def plot_9(patches):
    plt.figure()
    for i in range(3):
        for j in range(3):
            id = i*3+j+1
            plt.subplot(3, 3, id)
            plt.imshow(patches[id-1])
            plt.xticks([])
            plt.yticks([])
    plt.savefig("./data/predict_200_hist/test.png")


def main(csv_path, save_path):
    csv_data = parse_csv_data(csv_path)
    print(len(csv_data))

    def task(i, row):
        svs_img = svs_imgs[row['name']]
        patches = get_patches_with_center_point(svs_img, row['x'], row['y'])
        return i, patches, row['name']

    ts = []
    for i, row in enumerate(csv_data):
        ts.append(pool.submit(task, i, row))

    true = 0
    a = 0
    count = {}
    all = {}
    for t in tqdm(ts):
        t.done()
        i, patches, img_name = t.result()
        # plot_9(patches)
        preds = predict(patches)
        del patches
        # print(preds)
        csv_data[i]['cnn_preds'] = preds
        right = np.sum(csv_data[i]['gt'] ==
                       np.argmax(csv_data[i]['cnn_preds'], axis=-1))

        true += right
        a += 1
        if img_name not in count:
            count[img_name] = right
        else:
            count[img_name] += right

        if img_name not in all:
            all[img_name] = 1*9
        else:
            all[img_name] += 1*9

    print(true, a, true/(a*9))
    for img_name in count:
        r = count[img_name]/float(all[img_name])
        print(img_name+":"+str(r)+"," +
              str(count[img_name])+","+str(all[img_name]))

    with open(save_path, 'wb') as f:
        pickle.dump(csv_data, f, 3)


if __name__ == '__main__':
    base_name = "./data/rs_hist_inceptionResNetV2/%s"
    main("./data/origin/valid_point_200_hist.csv",
         base_name % "valid.pickle")
    main("./data/origin/test_point_200_hist.csv",
         base_name % "test.pickle")
    main("./data/origin/train_point_200_hist.csv",
         base_name % "train.pickle")
