import matplotlib.pyplot as plt
import rasterio
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
from core.wandb_logger import WandbLogger
import os
from data.metrics import PSNR, SSIM
from data.util import save_img, img_meta
import numpy as np
from data.sen12mscr import normalization


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/val.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Logger.setup_logger(None, opt['path']['log'],
    #                     'train', level=logging.INFO, screen=True)
    # Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    # logger.info(Logger.dict2str(opt))


    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    '''13通道预测'''
    # root = './dataset/val_dataset/s1/'
    # save_root = './dataset/results/13/'
    # for path in os.listdir(root):
    #     s1_path = os.path.join(root, path)
    #     s2_path = s1_path.replace('/s1', '/s2').replace('_s1', '_s2')
    #     s2_cloudy_path = s1_path.replace('/s1', '/s2_cloudy').replace('_s1', '_s2_cloudy')
    #     s1 = rasterio.open(s1_path).read()
    #     s2 = rasterio.open(s2_path).read()
    #     s2_cloudy = rasterio.open(s2_cloudy_path).read()
    #     s1 = normalization(s1)
    #     s2 = normalization(s2)
    #     s2_cloudy = normalization(s2_cloudy)
    #     data = {'path': path.replace('/s1', '/s2').replace('_s1', '_s2')}
    #     HR = np.concatenate([s1, s2], 0).astype(np.float16)
    #     SR = np.concatenate([s1, s2_cloudy], 0).astype(np.float16)
    #     data['HR'] = torch.from_numpy(HR.reshape(1, 15, 128, 128))
    #     data['SR'] = torch.from_numpy(SR.reshape(1, 15, 128, 128))
    #     diffusion.feed_data(data)
    #     diffusion.test(continous=True)
    #     visuals = diffusion.get_current_visuals()
    #     SR = np.array(visuals['SR'])[-1][5:2:-1, :, :].transpose(1, 2, 0)
    #     im = (SR - np.min(SR)) / (np.max(SR) - np.min(SR))
    #     plt.imsave(os.path.join(save_root, data['path']).replace('tif', 'png'), im)

    '''13通道预测，除rgb外，其余通道变为0'''
    root = './dataset/val_dataset/s1/'
    save_root = './dataset/results/13_000/'
    for path in os.listdir(root):
        s1_path = os.path.join(root, path)
        s2_path = s1_path.replace('/s1', '/s2').replace('_s1', '_s2')
        s2_cloudy_path = s1_path.replace('/s1', '/s2_cloudy').replace('_s1', '_s2_cloudy')
        s1 = rasterio.open(s1_path).read()
        s2 = rasterio.open(s2_path).read()
        s2_cloudy = rasterio.open(s2_cloudy_path).read()
        s1 = normalization(s1)
        s2 = normalization(s2)
        s2_cloudy = normalization(s2_cloudy)
        data = {'path': path.replace('/s1', '/s2').replace('_s1', '_s2')}
        HR = np.concatenate([s1, s2], 0).astype(np.float16)
        SR = np.concatenate([s1, s2_cloudy], 0).astype(np.float16)
        HR[0:3, :, :] = HR[0:3, :, :] * 0
        HR[6:, :, :] = HR[6:, :, :] * 0
        SR[0:3, :, :] = SR[0:3, :, :] * 0
        SR[6:, :, :] = SR[6:, :, :] * 0
        data['HR'] = torch.from_numpy(HR.reshape(1, 15, 128, 128))
        data['SR'] = torch.from_numpy(SR.reshape(1, 15, 128, 128))
        diffusion.feed_data(data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals()
        SR = np.array(visuals['SR'])[-1][5:2:-1, :, :].transpose(1,2,0)
        im = (SR - np.min(SR)) / (np.max(SR) - np.min(SR))
        plt.imsave(os.path.join(save_root, data['path']).replace('tif', 'png'), im)

    '''rgb三通道预测'''
    # root = './dataset/val_dataset/s1/'
    # save_root = './dataset/results/rgb/'
    # for path in os.listdir(root):
    #     s1_path = os.path.join(root, path)
    #     s2 = rasterio.open(s1_path.replace('/s1', '/s2').replace('_s1', '_s2')).read()
    #     s2_cloudy = rasterio.open(s1_path.replace('/s1', '/s2_cloudy').replace('_s1', '_s2_cloudy')).read()
    #     s2 = normalization(s2)
    #     s2_cloudy = normalization(s2_cloudy)
    #     data = {}
    #     data['path'] = path.replace('/s1', '/s2_cloudy').replace('_s1', '_s2_cloudy')
    #     data['HR'] = torch.from_numpy(s2[3:0:-1].copy().reshape(1, 3, 128, 128).astype(np.float16))
    #     data['SR'] = torch.from_numpy(s2_cloudy[3:0:-1].copy().reshape(1, 3, 128, 128).astype(np.float16))
    #
    #
    #     diffusion.feed_data(data)
    #     diffusion.test(continous=True)
    #     visuals = diffusion.get_current_visuals()
    #
    #     # single img series
    #     SR = np.array(visuals['SR'])[-1]
    #     im = ((SR - np.min(SR)) / (np.max(SR) - np.min(SR))).transpose(1,2,0)
    #     plt.imsave(os.path.join(save_root, data['path']).replace('tif', 'png'), im)
