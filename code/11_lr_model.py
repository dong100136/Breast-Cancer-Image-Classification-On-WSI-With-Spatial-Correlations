import numpy as np
from tqdm import tqdm
import openslide
from sklearn.linear_model import LogisticRegressionCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, log_loss
import cv2
import os

# config
model_name = "inceptionResNetV2"
base_path = "./data/rs_hist_%s" % model_name
csv_path = base_path+"/predict_map.csv"
origin_csv_path = "./data/origin/origin_200_hist.csv"
train_csv_path = "./data/origin/train_point_200_hist.csv"
valid_csv_path = "./data/origin/valid_point_200_hist.csv"
test_csv_path = "./data/origin/test_point_200_hist.csv"
# kernel = 1  # 3*3

svs_img_names = ['A01', 'A02', 'A03', 'A04',
                 'A05', 'A06', 'A07', 'A08',
                 "A09", "A10"]
svs_base_path = "./data/origin/%s/%s.svs"

svs_imgs = {svs_name: openslide.OpenSlide(svs_base_path % (
    svs_name, svs_name)) for svs_name in svs_img_names}

patch_size = 200

svs_size = {svs_name: [svs_imgs[svs_name].level_dimensions[0][1]//patch_size,
                       svs_imgs[svs_name].level_dimensions[0][0]//patch_size]
            for svs_name in svs_img_names
            }

feature_map = {
    svs_name: np.zeros([svs_size[svs_name][0], svs_size[svs_name][1], 4])
    for svs_name in svs_img_names
}

lr_imgs = {
    svs_name: np.zeros([svs_size[svs_name][0]*10, svs_size[svs_name][1]*10, 3])
    for svs_name in svs_img_names
}

color_maps = {
    0: np.array([255, 255, 255]),
    1: np.array([255, 0, 0]),
    2: np.array([0, 255, 0]),
    3: np.array([0, 0, 255])
}
# end config

# load feature map
with open(csv_path, 'r') as f:
    for line in f:
        img_name, x, y, label, preds1, preds2, preds3, preds4 = line.strip().split(',')

        x = int(x)//patch_size
        y = int(y)//patch_size
        label = int(label)
        preds = list(map(float, [preds1, preds2, preds3, preds4]))
        preds_label = np.argmax(preds, axis=-1)

        feature_map[img_name][x, y] = preds


def make_dataset(csv_path, kernel):
    rows = []
    with open(csv_path, 'r') as f:
        for line in f:
            img_name, x, y, c1, c2, c3, c4, c5, c6, c7, c8, c9 = line.strip().split(",")
            rows.append([img_name, int(x), int(y), int(c5)])

    x_train = np.zeros((len(rows), (2*kernel+1)*(2*kernel+1)*4))
    y_train = np.zeros((len(rows), 4))
    preds_0 = np.zeros((len(rows)))
    for k, row in tqdm(enumerate(rows)):
        img_name, x, y, c5 = row
        x = x//patch_size
        y = y//patch_size

        id = -1
        for i in range(-kernel, kernel+1):
            for j in range(-kernel, kernel+1):
                id += 1
                if x+i < svs_size[img_name][0] and y+j < svs_size[img_name][1]:
                    x_train[k][id*4:id*4+4] = feature_map[img_name][x+i, y+j]
                else:
                    pass
                    # print("out of bound")

        preds_0[k] = np.argmax(feature_map[img_name][x,y],axis=-1)
        y_train[k, c5] = 1

    return x_train, y_train, rows,preds_0


def eval(model, data_x, data_y, kernel, mode):
    data_predict = model.predict_proba(data_x)
    loss = log_loss(data_y, data_predict)
    acc = accuracy_score(np.argmax(data_y, axis=-1),
                         np.argmax(data_predict, axis=-1))
    center_point = 4*(kernel*kernel//2)
    pre_acc = accuracy_score(np.argmax(data_y, axis=-1),
                             np.argmax(data_x[:, center_point:center_point+4], axis=-1))
    print("%s pre acc:%.4f acc:%.4f loss:%.4f" % (mode, pre_acc, acc, loss))


def main(kernel=1):
    print("kernel size : %d" % (2*kernel+1))
    x_train, y_train, _,_ = make_dataset(train_csv_path, kernel)
    x_valid, y_valid, _,_ = make_dataset(valid_csv_path, kernel)
    x_test, y_test, _,_ = make_dataset(test_csv_path, kernel)

    reg = LogisticRegressionCV(cv=5, max_iter=200)
    model = OneVsRestClassifier(reg)
    model.fit(x_train, y_train)

    eval(model, x_train, y_train, kernel, 'train')
    eval(model, x_valid, y_valid, kernel, 'eval')
    eval(model, x_test, y_test, kernel, 'test')

    x_all, y_all, rows,preds_0 = make_dataset(origin_csv_path, kernel)
    preds_all = model.predict_proba(x_all)
    for row, pred in zip(rows, preds_all):
        img_name, x, y, _ = row
        pred_label = np.argmax(pred)

        x = (x-patch_size//2)//patch_size*10
        y = (y-patch_size//2)//patch_size*10

        lr_imgs[img_name][x:x+10, y:y+10] = color_maps[pred_label]

    for svs_name in svs_img_names:
        lr = lr_imgs[svs_name][:, :, -1::-1]
        cv2.imwrite(os.path.join(base_path, svs_name +
                                 "_lr_%d.png" % (2*kernel+1)), lr)

    return rows, preds_all,preds_0


rows, preds_all_1,preds_0 = main(kernel=1)
rows, preds_all_2,preds_0 = main(kernel=2)
rows, preds_all_3,preds_0 = main(kernel=3)
main(kernel=4)
main(kernel=5)

print(np.shape(preds_all_1))
print(np.shape(preds_all_2))
print(np.shape(preds_all_3))
print(np.shape(preds_0))

f = open(base_path+"/rs_all.csv",'w')
for (img_name, x, y, c5),pred_0, pred_1, pred_2, pred_3 in zip(rows,
                                                        preds_0,
                                                        np.argmax(
                                                            preds_all_1, axis=-1),
                                                        np.argmax(
                                                            preds_all_2, axis=-1),
                                                        np.argmax(
                                                            preds_all_3, axis=-1)):
    f.write("%s,%d,%d,%d,%d,%d,%d,%d\n"%(img_name,x,y,c5,pred_0,pred_1,pred_2,pred_3))

f.flush()
f.close()
