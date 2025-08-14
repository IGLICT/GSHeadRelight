import sys
import os

import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm

import legacy

from camera_utils import LookAtPoseSampler
from torch_utils import misc

from training.gs_generator import GSGenerator
import time
from custom_utils import save_ply

import json

import cv2
import pyshtools
from PIL import Image
from typing import List, Optional, Tuple, Union
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
import warnings
warnings.filterwarnings("ignore")




def to_tensor(img: Union[Image.Image, np.ndarray], normalize=True, device='cpu') -> torch.Tensor:
    if isinstance(img, Image.Image):
        img = np.array(img)
        if len(img.shape) > 2:
            img = img.transpose(2, 0, 1)
        else:
            img = img[None, ...]
    else:
        if img.shape[0] == img.shape[1]:
            img = img.transpose(2, 0, 1)
    if normalize:
        img = torch.from_numpy(img).to(torch.float32) / 127.5 - 1
    else:
        img = torch.from_numpy(img).to(torch.float32) / 255.
    return img[None, ...].to(device)

def shtools_matrix2vec(SH_matrix):
    numOrder = SH_matrix.shape[1]
    vec_SH = np.zeros(numOrder**2)
    count = 0
    for i in range(numOrder):
        for j in range(i,0,-1):
            vec_SH[count] = SH_matrix[1,i,j]
            count = count + 1
        for j in range(0,i+1):
            vec_SH[count]= SH_matrix[0, i,j]
            count = count + 1
    return vec_SH

def shtools_getSH(envMap, order=5):
    SH_r =  pyshtools.expand.SHExpandDH(envMap[...,0], sampling=2, lmax_calc=order, norm=4)
    SH_g =  pyshtools.expand.SHExpandDH(envMap[...,1], sampling=2, lmax_calc=order, norm=4)
    SH_b =  pyshtools.expand.SHExpandDH(envMap[...,2], sampling=2, lmax_calc=order, norm=4)
    return SH_r, SH_g, SH_b

def convert_env_to_img(env):
    im_gamma_correct = np.clip(np.power(env, 0.45), 0, 1)
    return to_tensor(Image.fromarray((im_gamma_correct*255).astype(np.uint8)))

def rotate_SH(SH, angles):
    """
    Rotate the SH coefficients.
    :param SH: SH coefficients matrix.
    :param angles: Rotation angles (alpha, beta, gamma) in degrees.
    :return: Rotated SH coefficients matrix.
    """
    alpha, beta, gamma = np.radians(angles)
    x = np.array([alpha, beta, gamma])
    dj = pyshtools.rotate.djpi2(SH.shape[-1])
    rotated_SH = pyshtools.rotate.SHRotateRealCoef(SH, x, dj)
    return rotated_SH

def get_SH_from_env(path_to_envMap: str, rotation_angles=(0, 0, 0), device='cpu'):
    if path_to_envMap.endswith('.exr'):
        env = imageio.imread(path_to_envMap, format='EXR-FI')
    else:
        env = imageio.imread(path_to_envMap)
    env = cv2.resize(env, (1024, 512))
    SH_r, SH_g, SH_b = shtools_getSH(env, 2)
    SH_r_rotated = rotate_SH(SH_r, rotation_angles)
    SH_g_rotated = rotate_SH(SH_g, rotation_angles)
    SH_b_rotated = rotate_SH(SH_b, rotation_angles)
    SH_matrix = np.vstack([SH_r_rotated[None], SH_g_rotated[None], SH_b_rotated[None]])
    SH = np.vstack([shtools_matrix2vec(SH_r_rotated)[None, ...],
                   shtools_matrix2vec(SH_g_rotated)[None, ...],
                   shtools_matrix2vec(SH_b_rotated)[None, ...]])
    # factor = (np.random.rand()*0.2 + 0.7)/SH.max()
    # factor = 1 / max(1, SH.max())
    # SH *= factor
    # return torch.from_numpy(SH).to(device), convert_env_to_img(env)
    return SH_matrix, SH

envmap_dir = '/home/jovyan/data7/lvhenglei/projects/example_hdr'

envmap_names = os.listdir(envmap_dir)
envmap_names.sort()
envmap_paths = [os.path.join(envmap_dir, envmap_name) for envmap_name in envmap_names]
matrices, shs = [], []
for envmap_path in envmap_paths:
    matrix, sh = get_SH_from_env(envmap_path, rotation_angles=(0,0,0))
    matrices.append(matrix)
    shs.append(sh)
