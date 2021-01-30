import cv2
import skimage
import torch
from torch import nn

from jpeg_utils import *


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


class GaussianNoise(nn.Module):
    def __init__(self, mean=0.0, var=0.01):
        super(GaussianNoise, self).__init__()
        self.mean, self.var = mean, var
    
    def forward(self, x):
        noise_x = None
        x = x.cpu().detach().numpy().transpose((0, 2, 3, 1))
        for i in range(x.shape[0]):
            xi = x[i]
            noise_xi = skimage.util.random_noise(xi, mode='gaussian', mean=self.mean, var=self.var)
            noise_xi = torch.from_numpy(noise_xi.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if i == 0:
                noise_x = noise_xi.unsqueeze(0)
            else:
                noise_x = torch.cat((noise_x, noise_xi.unsqueeze(0)), dim=0)
        return noise_x


class GaussianBlur(nn.Module):
    def __init__(self, size=3):
        super(GaussianBlur, self).__init__()
        self.size = size
    
    def forward(self, x):
        noise_x = None
        x = x.cpu().detach().numpy().transpose((0, 2, 3, 1))
        for i in range(x.shape[0]):
            xi = x[i]
            noise_xi = cv2.GaussianBlur(xi, (self.size, self.size), 0)
            noise_xi = torch.from_numpy(noise_xi.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if i == 0:
                noise_x = noise_xi.unsqueeze(0)
            else:
                noise_x = torch.cat((noise_x, noise_xi.unsqueeze(0)), dim=0)
        return noise_x


class Resize(nn.Module):
    def __init__(self, reh=64, rew=64):
        super(Resize, self).__init__()
        self.reh, self.rew = reh, rew
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.cpu().detach().numpy().transpose((0, 2, 3, 1))
        for i in range(x.shape[0]):
            xi = x[i]
            noise_xi = cv2.resize(cv2.resize(xi, (self.rew, self.reh)), (w, h))
            noise_xi = torch.from_numpy(noise_xi.transpose((2, 0, 1))).type(torch.FloatTensor).cuda()
            if i == 0:
                noise_x = noise_xi.unsqueeze(0)
            else:
                noise_x = torch.cat((noise_x, noise_xi.unsqueeze(0)), dim=0)
        return noise_x


class DiffJPEG(nn.Module):
    def __init__(self, h=128, w=128, differentiable=True, quality=50):
        super(DiffJPEG, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompress_jpeg(h, w, rounding=rounding, factor=factor)
    
    def forward(self, x):
        y, cb, cr = self.compress(x)
        return self.decompress(y, cb, cr)
