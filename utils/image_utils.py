# encoding: utf-8
"""
@author: monitor1379 
@contact: yy4f5da2@hotmail.com
@site: www.monitor1379.com

@version: 1.0
@license: Apache Licence
@file: image_utils.py
@time: 2016/9/25 15:04


"""

import numpy as np


def hwd2dhw(img):
    new_img = np.empty((img.shape[2], img.shape[0], img.shape[1]))
    for d in xrange(img.shape[2]):
        new_img[d] = img[:, :, d]
    return new_img


def batch_hwd2dhw(imgs):
    new_imgs = np.empty((imgs.shape[0], imgs.shape[3], imgs.shape[1], imgs.shape[2]))
    for i in xrange(imgs.shape[0]):
        new_imgs[i] = hwd2dhw(imgs[i])
    return new_imgs


def dhw2hwd(img):
    new_img = np.empty((img.shape[1], img.shape[2], img.shape[0]))
    for d in xrange(img.shape[0]):
        new_img[:, :, d] = img[d]
    return new_img

def batch_dhw2hwd(imgs):
    new_imgs = np.empty((imgs.shape[0], imgs.shape[2], imgs.shape[3], imgs.shape[1]))
    for i in xrange(imgs.shape[0]):
        new_imgs[i] = dhw2hwd(imgs[i])
    return new_imgs