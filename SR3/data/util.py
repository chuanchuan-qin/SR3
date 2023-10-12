import os

import numpy as np
import rasterio


def img_meta(tif_path):
    img = rasterio.open(tif_path)
    return img.meta


def read_img(tif_path):
    img = rasterio.open(tif_path)
    data = img.read()
    return data.astype(np.float16)


def save_img(tif_save_path, img, meta):
    os.makedirs(tif_save_path.replace(tif_save_path.split('/')[-1], ''), exist_ok=True)
    with rasterio.open(tif_save_path, 'w', driver=meta['driver'], height=meta['height'], width=meta['width'],
                       count=meta['count'], transform=meta['transform'], crs=meta['crs'],
                       dtype=meta['dtype']) as tif:
        tif.write(img)
        tif.close()


if __name__ == '__main__':
    s1 = read_img('C:/Users/chuanchuan/Desktop/s1_1/ROIs1158_spring_s1_1_p30.tif')
    # s2 = read_img('C:/Users/chuanchuan/Desktop/s2_1/ROIs1158_spring_s2_1_p30.tif')
    # s2_cloudy = read_img('C:/Users/chuanchuan/Desktop/s2_cloudy_1/ROIs1158_spring_s2_cloudy_1_p30.tif')
    # meta_s1 = img_meta('C:/Users/chuanchuan/Desktop/s1_1/ROIs1158_spring_s1_1_p30.tif')
    # meta_s2 = img_meta('C:/Users/chuanchuan/Desktop/s2_1/ROIs1158_spring_s2_1_p30.tif')
    # meta_s2_cloudy = img_meta('C:/Users/chuanchuan/Desktop/s2_cloudy_1/ROIs1158_spring_s2_cloudy_1_p30.tif')
    # print(f's1.shape:{s1.shape}, s2.shape:{s2.shape}, s2_cloudy.shape:{s2_cloudy.shape}')
    # save_img('C:/Users/chuanchuan/Desktop/test_s1.tif', s1, meta_s1)
    test = read_img('C:/Users/chuanchuan/Desktop/test_s1.tif')
    # x = torch.ones([2, 256, 256], dtype=torch.float64)
    # x.astype(np.float64)
    # print(type(test))
    print(np.var(test/10000))