import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms, datasets

import torch.optim as optim
import gc

from utils import *
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from datetime import datetime
import argparse

import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import math
import os.path
import pandas as pd
# from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
from pandas import DataFrame
from datetime import datetime
import random
import numpy as np
import os
from net.Swin_Encoder import Swin_Encoder


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import argparse



class config():
    image_dims = (3, 256, 256)
    encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
            C=32, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )

encoder = Swin_Encoder( config).to("cuda")#加载还差点初始化的东西，
batch_size = 1
channels = 3
height = 256
width = 256
    
    # 使用 torch.randn 创建一个符合标准正态分布的随机张量
random_image_256 = torch.randn(batch_size, channels, height, width).to("cuda")
print(encoder(random_image_256).shape)