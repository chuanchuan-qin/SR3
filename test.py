import random
import subprocess
import rasterio
import os
import numpy as np
import torch
import tqdm
from data.util import img_meta, save_img, read_img
from data.sen12mscr import SEN12MSCR
from torch.utils.data import DataLoader


def cut_tif(path):
    tif = rasterio.open(path)
    img = tif.read()
    meta = img_meta(path)
    meta['width'] = 128
    meta['height'] = 128
    img_1 = img[:, 0:128, 0:128]
    save_img(path.replace('sen12mscr', 'new_sen12mscr').replace('.tif', '_1.tif'), img_1, meta)
    img_2 = img[:, 0:128, 128:256]
    save_img(path.replace('sen12mscr', 'new_sen12mscr').replace('.tif', '_2.tif'), img_2, meta)
    img_3 = img[:, 128:256, 0:128]
    save_img(path.replace('sen12mscr', 'new_sen12mscr').replace('.tif', '_3.tif'), img_3, meta)
    img_4 = img[:, 128:256, 128:256]
    save_img(path.replace('sen12mscr', 'new_sen12mscr').replace('.tif', '_4.tif'), img_4, meta)


def get_RGB(path):
    pass


# dataset = SEN12MSCR()
# paths = dataset.paths
# val_paths = random.sample(paths, 2000)
# print(len(dataset.paths))
#
#
# '''随机取2000张当作验证数据集'''
# for i in val_paths:
#     command = 'mv '+i['S1']+' /data/zy/qcc/SR3/dataset/val_dataset/s1/'
#     subprocess.run(command, shell=True)
#     command = 'mv ' + i['S2'] + ' /data/zy/qcc/SR3/dataset/val_dataset/s2/'
#     subprocess.run(command, shell=True)
#     command = 'mv '+i['S2_cloudy']+' /data/zy/qcc/SR3/dataset/val_dataset/s2_cloudy/'
#     subprocess.run(command, shell=True)
# print(len(dataset.paths))

s1_max, s1_min = 0, 0
s2_max, s2_min = 0, 0
s2_cloudy_max, s2_cloudy_min = 0, 0
dataset = SEN12MSCR()
paths = dataset.paths
j = 0
for path in tqdm.tqdm(paths):
    s1 = rasterio.open(path['S1']).read()
    s2 = rasterio.open(path['S2']).read()
    s2_cloudy = rasterio.open(path['S2_cloudy']).read()
    s1_max, s1_min = max(s1_max, np.max(s1)), min(s1_min, np.min(s1))
    s2_max, s2_min = max(s2_max, np.max(s2)), min(s2_min, np.min(s2))
    s2_cloudy_max, s2_cloudy_min = max(s2_cloudy_max, np.max(s2_cloudy)), min(s2_cloudy_min, np.min(s2_cloudy))
    j += 1
    if j % 1000 == 0:
        print("-----------")
        print(s1_max, s1_min)
        print(s2_max, s2_min)
        print(s2_cloudy_max, s2_cloudy_min)
        print("-----------")

# for path in paths:
#     for _, p in path.items():
#         p1 = p.replace('sen12mscr', 'new_sen12mscr').replace('.tif', '_1.tif')
#         p2 = p.replace('sen12mscr', 'new_sen12mscr').replace('.tif', '_2.tif')
#         p3 = p.replace('sen12mscr', 'new_sen12mscr').replace('.tif', '_3.tif')
#         p4 = p.replace('sen12mscr', 'new_sen12mscr').replace('.tif', '_4.tif')
#         # print(p)
#         # print(p1)
#         # print(p2)
#         # print(p3)
#         # print(p4)
#         meta = img_meta(p)
#         meta['width'] = 128
#         meta['height'] = 128
#         img = read_img(p)
#         # print(meta)
#         img1 = img[:, 0:128, 0:128]
#         img2 = img[:, 0:128, 128:256]
#         img3 = img[:, 128:256, 0:128]
#         img4 = img[:, 128:256, 128:256]
#         # print(img1.shape, img2.shape, img3.shape, img4.shape)
#         save_img(p1, img1, meta)
#         save_img(p2, img2, meta)
#         save_img(p3, img3, meta)
#         save_img(p4, img4, meta)

# S1_max = -100
# S2_max = -100
# S2_cloudy_max = -100
# S1_min = 1000
# S2_min = 1000
# S2_cloudy_min = 1000
# dataset = SEN12MSCR()
# paths = dataset.paths
# S1_mean = 0.
# S2_mean = 0.
# S2_cloudy_mean = 0.
# S1_var = 0.
# S2_var = 0.
# S2_cloudy_var = 0.
# i = 1
# print(len(paths))
# for path in paths:
#     S1 = read_img(path['S1'])
#     S2 = read_img(path['S2'])
#     S2_cloudy = read_img(path['S2_cloudy'])
#
#     S1_max = max(S1_max, S1.max())
#     S2_max = max(S2_max, S2.max())
#     S2_cloudy_max = max(S2_cloudy_max, S2_cloudy.max())
#
#     S1_min = min(S1_min, S1.min())
#     S2_min = min(S2_min, S2.min())
#     S2_cloudy_min = min(S2_cloudy_min, S2_cloudy.min())
#     print(f'S1:{S1_max,S1_min},S2:{S2_max,S2_min},S2_cloudy:{S2_cloudy_max,S2_cloudy_min}')
#
#     S1_mean = (S1_mean * (i - 1) + S1.mean()) / i
#     S2_mean = (S2_mean * (i - 1) + S2.mean()) / i
#     S2_cloudy_mean = (S2_cloudy_mean * (i - 1) + S2_cloudy.mean()) / i
#     print(f'S1_mean:{S1_mean}, S2_mean:{S2_mean}, S2_cloudy_mean:{S2_cloudy_mean}')
#
#     # S1_var = (S1_var * (i - 1) + np.var(S1)) / i
#     # S2_var = (S2_var * (i - 1) + np.var(S2)) / i
#     # S2_cloudy_var = (S2_cloudy_var * (i - 1) + np.var(S2_cloudy)) / i
#     # print(f'S1_var:{S1_var}, S2_var:{S2_var}, S2_cloudy_var:{S2_cloudy_var}')
#     #

#
# img = rasterio.open(r'C:\Users\chuanchuan\Desktop\sen12mscr\ROIs1158_spring_s1\s1_1\ROIs1158_spring_s1_1_p30.tif').read()
# img1 = rasterio.open(r'C:\Users\chuanchuan\Desktop\new_sen12mscr\ROIs1158_spring_s1\s1_1\ROIs1158_spring_s1_1_p30_1.tif').read()
# print((img1 == img[:, 0:128, 0:128]).all())


