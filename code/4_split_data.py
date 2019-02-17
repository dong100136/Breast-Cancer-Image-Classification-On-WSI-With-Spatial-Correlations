import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import os

# config
train_img = ['A03', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10']
test_img = ['A01', 'A02']
origin_csv_path = "./data/origin/origin_200.csv"
img_size = 200
# end config

# header = ['name','x','y','c1','c2','c3','c4','c5','c6','c7','c8','c9']
# data = pd.read_csv("./data/origin/origin.csv",header=None)
# data.columns=header

train_data = [[], [], [], []]
valid_data = []
test_data = [[], [], [], []]

with open(origin_csv_path, 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)

    for line in tqdm(lines):
        img_name, x, y, c1, c2, c3, c4, c5, c6, c7, c8, c9 = line.split(',')
        if img_name in train_img:
            train_data[int(c5)].append(line)
        elif img_name in test_img:
            test_data[int(c5)].append(line)


def save_data(data, path):
    with open(path, 'w') as f:
        for d in data:
            f.write(d)


data_num = [len(x) for x in train_data]
min_data_num = np.min(data_num)
train_data_num = int(min_data_num*0.1)

train_data_path = "./data/origin/train_point_"+str(img_size)+".csv"
valid_data_path = "./data/origin/valid_point_"+str(img_size)+".csv"
test_data_path = "./data/origin/test_point_"+str(img_size)+".csv"
os.system("rm %s %s %s" %
          (train_data_path, valid_data_path, test_data_path))

new_train_data = [x[train_data_num:min_data_num] for x in train_data]
# new_train_data = [x[int(len(x)*0.1):] for x in train_data]
new_train_data = np.concatenate(new_train_data)
random.shuffle(new_train_data)
save_data(new_train_data, train_data_path)
print("train:"+str(len(new_train_data)))

new_valid_data = [x[:train_data_num] for x in train_data]
# new_valid_data = [x[:int(len(x)*0.1)] for x in train_data]
new_valid_data = np.concatenate(new_valid_data)
random.shuffle(new_valid_data)
save_data(new_valid_data, valid_data_path)
print("valid:"+str(len(new_valid_data)))

data_num = [len(x) for x in test_data]
min_data_num = np.min(data_num)

# new_test_data = test_data
new_test_data = [x[:min_data_num] for x in test_data]
new_test_data = np.concatenate(new_test_data)
random.shuffle(new_test_data)
save_data(new_test_data, test_data_path)
print("test:"+str(len(new_test_data)))
