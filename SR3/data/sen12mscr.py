import os

import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset
from data.util import read_img


def normalization(img):
    if len(img) == 2:
        clip_min = -25.0
        clip_max = 0.0
    elif len(img) == 13:
        clip_min = 0
        clip_max = 10000
    else:
        print('通道数错误！')
        exit()
    img = np.clip(a=img, a_min=clip_min, a_max=clip_max)
    img = img / (clip_max - clip_min)
    return img


""" 
    SEN12MS-CR dataset class, inherits from torch.utils.data.Dataset

    INPUT: 
    root:               str, path to the SEN12MS-CR-TS data set
    phase:              str, in ['train', 'val', 'test']
"""


class SEN12MSCR(Dataset):

    def __init__(self, root='dataset/new_sen12mscr/', phase="train"):
        self.root_dir = root
        self.phase = phase
        assert self.phase in ['train', 'val', 'test'], "phase should in ['train', 'val', 'test']"
        if self.phase == 'train':
            self.paths = self.get_train_paths()
        elif self.phase == 'val':
            self.paths = self.get_val_paths()
        elif self.phase == 'test':
            self.paths = self.get_test_paths()

        self.n_samples = len(self.paths)

    def get_val_paths(self):
        pass

    def get_test_paths(self):
        print(f'Processing paths for {self.phase} dataset')
        test_root = './dataset/val_dataset/s1/'
        print(test_root)
        paths = []
        print(len(os.listdir(test_root)))
        for path in os.listdir(test_root):
            paths_S1 = os.path.join(test_root, path)
            paths_S2 = paths_S1.replace('/s1', '/s2').replace('_s1', '_s2')
            paths_S2_cloudy = paths_S1.replace('/s1', '/s2_cloudy').replace('_s1', '_s2_cloudy')

            sample = {"S1": paths_S1,
                      "S2": paths_S2,
                      "S2_cloudy": paths_S2_cloudy}
            paths.append(sample)
        return paths

    def get_train_paths(self):
        print(f'Processing paths for {self.phase} dataset')
        paths = []
        seeds_S1 = natsorted([s1dir for s1dir in os.listdir(self.root_dir) if "_s1" in s1dir])
        for seed in seeds_S1:
            rois_S1 = natsorted(os.listdir(os.path.join(self.root_dir, seed)))
            for roi in rois_S1:
                roi_dir = os.path.join(self.root_dir, seed, roi)
                # if os.path.join(seed, roi) not in self.dirs:
                #     continue
                paths_S1 = natsorted([os.path.join(roi_dir, s1patch) for s1patch in os.listdir(roi_dir)])
                paths_S2 = [patch.replace('/s1', '/s2').replace('_s1', '_s2') for patch in paths_S1]
                paths_S2_cloudy = [patch.replace('/s1', '/s2_cloudy').replace('_s1', '_s2_cloudy') for patch in
                                   paths_S1]

                for pdx in range(len(paths_S1)):
                    sample = {"S1": paths_S1[pdx],
                              "S2": paths_S2[pdx],
                              "S2_cloudy": paths_S2_cloudy[pdx]}
                    paths.append(sample)
        return paths

    def __getitem__(self, pdx):  # get the triplet of patch with ID pdx
        s1 = normalization(read_img(self.paths[pdx]['S1']))
        s2 = normalization(read_img(self.paths[pdx]['S2']))
        s2_cloudy = normalization(read_img(self.paths[pdx]['S2_cloudy']))
        HR = np.concatenate([s1, s2], 0)
        SR = np.concatenate([s1, s2_cloudy], 0)

        '''除rgb外， 其余通道换为0'''
        # HR[0:3, :, :] = HR[0:3, :, :] * 0
        # HR[6:, :, :] = HR[6:, :, :] * 0
        # SR[0:3, :, :] = SR[0:3, :, :] * 0
        # SR[6:, :, :] = SR[6:, :, :] * 0

        '''rgb三通道'''
        # HR = HR[3:6, :, :].copy()
        # SR = SR[3:6, :, :].copy()

        path = self.paths[pdx]['S2']
        return {'HR': HR, 'SR': SR, 'path': path}

    def __len__(self):
        # length of generated list
        return self.n_samples
