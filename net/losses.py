import torch
from torch import nn
import numpy as np
from .layers import bn, VarianceLayer, CovarianceLayer, GrayscaleLayer
from .downsampler import * 
from torch.nn import functional

import torch.nn.functional as F
from torch.autograd import Variable

from math import exp




class StdLoss(nn.Module):
    def __init__(self):
        """
        Loss on the variance of the image.
        Works in the grayscale.
        If the image is smooth, gets zero
        """
        super(StdLoss, self).__init__()
        blur = (1 / 25) * np.ones((5, 5))
        blur = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.mse = nn.MSELoss()
        self.blur = nn.Parameter(data=torch.cuda.FloatTensor(blur), requires_grad=False)
        image = np.zeros((5, 5))
        image[2, 2] = 1
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        self.image = nn.Parameter(data=torch.cuda.FloatTensor(image), requires_grad=False)
        self.gray_scale = GrayscaleLayer()

    def forward(self, x):
        x = self.gray_scale(x)
        return self.mse(functional.conv2d(x, self.image), functional.conv2d(x, self.blur))


class ExclusionLoss(nn.Module):

    def __init__(self, level=3):
        """
        Loss on the gradient. based on:
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single_Image_Reflection_CVPR_2018_paper.pdf
        """
        super(ExclusionLoss, self).__init__()
        self.level = level
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2).type(torch.cuda.FloatTensor)
        self.sigmoid = nn.Sigmoid().type(torch.cuda.FloatTensor)

    def get_gradients(self, img1, img2):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            # alphax = 2.0 * torch.mean(torch.abs(gradx1)) / torch.mean(torch.abs(gradx2))
            # alphay = 2.0 * torch.mean(torch.abs(grady1)) / torch.mean(torch.abs(grady2))
            alphay = 1
            alphax = 1
            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1

            # gradx_loss.append(torch.mean(((gradx1_s ** 2) * (gradx2_s ** 2))) ** 0.25)
            # grady_loss.append(torch.mean(((grady1_s ** 2) * (grady2_s ** 2))) ** 0.25)
            gradx_loss += self._all_comb(gradx1_s, gradx2_s)
            grady_loss += self._all_comb(grady1_s, grady2_s)
            img1 = self.avg_pool(img1)
            img2 = self.avg_pool(img2)
        return gradx_loss, grady_loss

    def _all_comb(self, grad1_s, grad2_s):
        v = []
        for i in range(grad2_s.shape[1]):
            for j in range(grad1_s.shape[1]):
                v.append(torch.mean(((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2))) ** 0.25)
        return v

    def forward(self, img1, img2):
        gradx_loss, grady_loss = self.get_gradients(img1, img2)
        loss_gradxy = sum(gradx_loss) / (self.level * 9) + sum(grady_loss) / (self.level * 9)
        return loss_gradxy / 2.0

    def compute_gradient(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady


class ExtendedL1Loss(nn.Module):
    """
    also pays attention to the mask, to be relative to its size
    """
    def __init__(self):
        super(ExtendedL1Loss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, a, b, mask):
        normalizer = self.l1(mask, torch.zeros(mask.shape).cuda())
        # if normalizer < 0.1:
        #     normalizer = 0.1
        c = self.l1(mask * a, mask * b) / normalizer
        return c


class NonBlurryLoss(nn.Module):
    def __init__(self):
        """
        Loss on the distance to 0.5
        """
        super(NonBlurryLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, x):
        return 1 - self.mse(x, torch.ones_like(x) * 0.5)


class GrayscaleLoss(nn.Module):
    def __init__(self):
        super(GrayscaleLoss, self).__init__()
        self.gray_scale = GrayscaleLayer()
        self.mse = nn.MSELoss().cuda()

    def forward(self, x, y):
        x_g = self.gray_scale(x)
        y_g = self.gray_scale(y)
        return self.mse(x_g, y_g)


class GrayLoss(nn.Module):
    def __init__(self):
        super(GrayLoss, self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, x):
        y = torch.ones_like(x) / 2.
        return 1 / self.l1(x, y)


class GradientLoss(nn.Module):
    """
    L1 loss on the gradient of the picture
    """
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, a):
        gradient_a_x = torch.abs(a[:, :, :, :-1] - a[:, :, :, 1:])
        gradient_a_y = torch.abs(a[:, :, :-1, :] - a[:, :, 1:, :])
        return torch.mean(gradient_a_x) + torch.mean(gradient_a_y)


class YIQGNGCLoss(nn.Module):
    def __init__(self, shape=5):
        super(YIQGNGCLoss, self).__init__()
        self.shape = shape
        self.var = VarianceLayer(self.shape, channels=1)
        self.covar = CovarianceLayer(self.shape, channels=1)

    def forward(self, x, y):
        if x.shape[1] == 3:
            x_g = rgb_to_yiq(x)[:, :1, :, :]  # take the Y part
            y_g = rgb_to_yiq(y)[:, :1, :, :]  # take the Y part
        else:
            assert x.shape[1] == 1
            x_g = x  # take the Y part
            y_g = y  # take the Y part
        c = torch.mean(self.covar(x_g, y_g) ** 2)
        vv = torch.mean(self.var(x_g) * self.var(y_g))
        return c / vv

def tv_loss(labels,TV_WEIGHT):
    diff_i = torch.sum(torch.abs(labels[:, :, :, 1:] - labels[:, :, :, :-1]))
    diff_j = torch.sum(torch.abs(labels[:, :, 1:, :] - labels[:, :, :-1, :]))
    tv_loss = TV_WEIGHT * (diff_i + diff_j)
    return tv_loss.item()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

class MS_SSIM(torch.nn.Module):
    def __init__(self, size_average = True, max_val = 255):
        super(MS_SSIM, self).__init__()
        self.size_average = size_average
        self.channel = 1
        self.max_val = max_val
    def _ssim(self, img1, img2, size_average = True):

        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11
        window = create_window(window_size, sigma, self.channel).cuda()
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = self.channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = self.channel) - mu1_mu2

        C1 = (0.01*self.max_val)**2
        C2 = (0.03*self.max_val)**2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2*mu1_mu2 + C1)*V1)/((mu1_sq + mu2_sq + C1)*V2)
        mcs_map = V1 / V2
        if size_average:
            return ssim_map.mean(), mcs_map.mean()

    def ms_ssim(self, img1, img2, levels=5):

        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).cuda())

        msssim = Variable(torch.Tensor(levels,).cuda())
        mcs = Variable(torch.Tensor(levels,).cuda())
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2

        value = (torch.prod(mcs[0:levels-1]**weight[0:levels-1])*
                                    (msssim[levels-1]**weight[levels-1]))
        return value


    def forward(self, img1, img2):

        return self.ms_ssim(img1, img2)




def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)