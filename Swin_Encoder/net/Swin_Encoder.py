
from Swin_Encoder.net.encoder import *
from random import choice
import torch.nn as nn


class Swin_Encoder(nn.Module):
    def __init__(self, config):
        super(Swin_Encoder, self).__init__()
        self.config = config
        encoder_kwargs = config.encoder_kwargs
        self.encoder = create_encoder(**encoder_kwargs)


        self.H = self.W = 0
      
    def forward(self, input_image):
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.H = H
            self.W = W
        feature = self.encoder(input_image)
        return feature
