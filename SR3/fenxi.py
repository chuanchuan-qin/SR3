from data.sen12mscr import SEN12MSCR
from tqdm import tqdm

import matplotlib.pyplot as plt
import cv2
import numpy as np
import rasterio

# 参数设置
IMG_SIZE = (128, 128)
# 读取图片并生成数据集类，如果不用tf的方式读取数据集，可以自己定义任何读取方式，只需要替换第25和26行的image_dataset即可
dataset = SEN12MSCR()
file_paths_list = dataset.paths
hist_s1 = np.zeros((26, 2))
hist_s2 = np.zeros((28005, 13))
hist_s2_cloudy = np.zeros((28005, 13))
num = len(file_paths_list)
# 保存结果
result = {}
color = "red"
j = 0

'''
    s1: 5.2362084 -56.140903
    s2: 28003 0
    s2_cloudy: 28005 0
'''
for path in tqdm(file_paths_list):
    s1 = rasterio.open(path['S1']).read()
    s2 = rasterio.open(path['S2']).read()
    s2_cloudy = rasterio.open(path['S2_cloudy']).read()
    # print(s1.max(), s1.min(), s2.max(), s2.min(), s2_cloudy.max(), s2_cloudy.min())
    for i in range(len(s1)):
        now_hist_s1 = cv2.calcHist([s1[i]], [0], None, [26], [-25, 1]).reshape((26,))
        hist_s1[:, i] += now_hist_s1
    for i in range(len(s2)):
        now_hist_s2 = cv2.calcHist([s2[i]], [0], None, [28005], [0, 28005]).reshape((28005,))
        hist_s2[:, i] += now_hist_s2
    for i in range(len(s2_cloudy)):
        now_hist_s2_cloudy = cv2.calcHist([s2_cloudy[i]], [0], None, [28005], [0, 28005]).reshape((28005,))
        hist_s2_cloudy[:, i] += now_hist_s2_cloudy
    j += 1
    # print(hist_s1.shape, hist_s2.shape, hist_s2_cloudy.shape)
    if j % 100 == 0:
        result["s1"] = hist_s1
        result["s2"] = hist_s2
        result["s2_cloudy"] = hist_s2_cloudy
        for k, v in result.items():
            for i in range(v.shape[1]):
                plt.clf()
                plt.plot(v[1:, i], color=color)
                if k == 's1':
                    plt.xlim([-25, 0])
                else:
                    plt.xlim([0, 28005])
                plt.savefig('./hist/hist_' + k + '_' + str(i) + '.png', format='png')

