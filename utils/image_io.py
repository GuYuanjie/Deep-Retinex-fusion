import glob

import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
#import skvideo.io
import pywt
matplotlib.use('agg')

import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import cv2
# -*- coding: utf-8 -*-
import os
from skimage.measure import shannon_entropy
def gradient(x):
    kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=x.dtype, device='cuda:0')
    grad = torch.nn.functional.conv2d(x, kernel, padding=1)
    return grad

def cmp_AG(img):
    img = img * 255
    gradient_a_x = abs(img[ :, :, :-1] - img[ :, :, 1:])
    gradient_a_y = abs(img[ :, :-1, :] - img[ :, 1:, :])
    return np.mean(gradient_a_x) + np.mean(gradient_a_y)

def EN(img):
    return shannon_entropy(img)


def SD(img):
    return np.std(img)


def cross_covariance(x, y, mu_x, mu_y):
    return 1 / (x.size - 1) * np.sum((x - mu_x) * (y - mu_y))


def SSIM(x, y):
    L = np.max(np.array([x, y])) - np.min(np.array([x, y]))
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sig_x = np.std(x)
    sig_y = np.std(y)
    sig_xy = cross_covariance(x, y, mu_x, mu_y)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    C3 = C2 / 2
    return (2 * mu_x * mu_y + C1) * (2 * sig_x * sig_y + C2) * (sig_xy + C3) / (
            (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2) * (sig_x * sig_y + C3))


def correlation_coefficients(x, y):
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    return np.sum((x - mu_x) * (y - mu_y)) / np.sqrt(np.sum((x - mu_x) ** 2) * np.sum((y - mu_y) ** 2))


def CC(ir, vi, fu):
    rx = correlation_coefficients(ir, fu)
    ry = correlation_coefficients(vi, fu)
    return (rx + ry) / 2


def SF(I):
    I = I.astype(np.int16)
    RF = np.diff(I, 1, 0)
    # RF[RF < 0] = 0
    RF = RF ** 2
    # RF[RF > 255] = 255
    RF = np.sqrt(np.mean(RF))

    CF = np.diff(I, 1, 1)
    # CF[CF < 0] = 0
    CF = CF ** 2
    # CF[CF > 255] = 255
    CF = np.sqrt(np.mean(CF))
    return np.sqrt(RF ** 2 + CF ** 2)

def joint_grad(x_f, x_1, x_2):
    grad_1 = torch.abs(gradient(x_1))
    grad_2 = torch.abs(gradient(x_2))
    grad_f = torch.abs(gradient(x_f))
    grad_j, _ = torch.max(torch.cat((grad_1, grad_2), dim=1), dim=1, keepdim=True)
    return grad_j, grad_f, grad_2

def joint_grad_triple(x_f, x_1, x_2,x_3):
    grad_1 = torch.abs(gradient(x_1))
    grad_2 = torch.abs(gradient(x_2))
    grad_3 = torch.abs(gradient(x_3))
    grad_f = torch.abs(gradient(x_f))
    grad_j, _ = torch.max(torch.cat((grad_1, grad_2, grad_3), dim=1), dim=1, keepdim=True)
    return grad_j, grad_f

def listdir(path, list_name):  #传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)

def search_largest_region(image):
    labeling = measure.label(image)
    regions = measure.regionprops(labeling)

    largest_region = None
    area_max = 0.
    for region in regions:
        if region.area > area_max:
            area_max = region.area
            largest_region = region

    return largest_region


def generate_largest_region(image):
    region = search_largest_region(image)
    bin_image = np.zeros_like(image)
    for coord in region.coords:
        bin_image[coord[0], coord[1]] = 1
    return bin_image


def fillHole(im_in):
    im_floodfill = im_in.copy().astype(np.uint8)
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = (im_in + im_floodfill_inv) / 255
    im_out[im_out > 1] = 1
    #plt.imshow(im_out)
    #plt.show()
    return im_out.astype(int)

def rgb2y_CWH_nol(x):
    return  0.257 * x[0, :, :] + 0.504 * x[1, :, :] + 0.098 * x[2, :, :] + 0.0625

def rgb2_gray_nol(x):
    return  0.2126 * x[0, :, :] + 0.7152 * x[1, :, :] + 0.0722 * x[2, :, :]

def to_bin(x,k=0.5):
    v = torch.zeros_like(x)
    v[x > k] = 1
    return v
def reg_kernel(x):
    x[x < 0.2] = 1
    x[0.2<x <= 0.4] = 3
    x[0.4<x <= 0.6] = 5
    x[0.6<x <= 0.8] = 7
    x[0.8<x <= 1] = 9
    return x

def clear_segmetation(x,k=3,k_dilated=3,k_eroded=3,threshold=0.018):
    x1 = cv2.GaussianBlur(x, ksize=(k, k,), sigmaX=0, sigmaY=0)
    x2 = abs(x - x1)
    kernel_dilated = cv2.getStructuringElement(cv2.MORPH_RECT, (k_dilated, k_dilated))
    print(kernel_dilated)
    kernel_eroded = cv2.getStructuringElement(cv2.MORPH_RECT, (k_eroded, k_eroded))
    #dilated =
    #eroded = cv2.erode(dilated, kernel_eroded)
    ret, thresh1 = cv2.threshold(x2, threshold, 1, cv2.THRESH_BINARY)

    thresh1 = cv2.dilate(thresh1, kernel_dilated)
    thresh1 = cv2.erode(thresh1, kernel_eroded)


    #thresh1=cv2.erode(thresh1, kernel_eroded)

    #thresh1 = cv2.GaussianBlur(thresh1, ksize=(5, 5,), sigmaX=0, sigmaY=0)
    #thresh1= cv2.medianBlur(thresh1, 5)
    """
    """
    plt.subplot(1, 2, 1)
    plt.title("Soucre Image")
    plt.imshow(x2, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Segamenting Image")
    plt.imshow(thresh1)
    plt.show()

    return thresh1

def label2mask(x):
    """
    Input is
    """
    c = x.shape[0]
    h = x.shape[1]
    x1 = torch.zeros([1,1,c,h])
    x2 = torch.zeros_like(x1)
    x3 = torch.zeros_like(x1)
    for i in range(c):
        for j in range(h):
            if x[i,j]==0: x1[0,0,i,j]=1
            if x[i,j] == 1: x2[0, 0, i, j] = 1
            if x[i,j] == 2: x3[0, 0, i, j] = 1
    return x1,x2,x3

def to_DFWT(x):
    wtfcA = np.zeros([x.shape[0],x.shape[1],x.shape[2]//2,x.shape[3]//2]).astype(np.float32)
    wtfcH = np.zeros([x.shape[0],x.shape[1],x.shape[2]//2,x.shape[3]//2]).astype(np.float32)
    wtfcV = np.zeros([x.shape[0],x.shape[1],x.shape[2]//2,x.shape[3]//2]).astype(np.float32)
    wtfcD = np.zeros([x.shape[0],x.shape[1],x.shape[2]//2,x.shape[3]//2]).astype(np.float32)
    B = x.shape[0]
    C = x.shape[1]
    for batch in range(B):
        for channels in range(C):
            coeffs = pywt.dwt2(x[batch, channels, :, :].data.cpu().numpy(), 'haar')
            wtfcA[batch, channels, :, :], (wtfcH[batch, channels, :, :], wtfcV[batch, channels, :, :], wtfcD[batch, channels, :, :]) = coeffs
    return torch.from_numpy(wtfcA).type(torch.cuda.FloatTensor), torch.from_numpy(wtfcH).type(torch.cuda.FloatTensor), torch.from_numpy(wtfcV).type(torch.cuda.FloatTensor), torch.from_numpy(wtfcD).type(torch.cuda.FloatTensor)


def to_iDFWT(wtfcA, wtfcH, wtfcV, wtfcD):
    wtfcA = wtfcA.data.cpu().numpy()
    wtfcH = wtfcA.data.cpu().numpy()
    wtfcV = wtfcA.data.cpu().numpy()
    wtfcD = wtfcA.data.cpu().numpy()
    B = wtfcA.shape[0]
    C = wtfcA.shape[1]
    x = []
    for batch in range(B):
        for channels in range(C):
            coeffs = wtfcA[batch, channels, :, :],(wtfcH[batch, channels, :, :],wtfcV[batch, channels, :, :],wtfcD[batch, channels, :, :])
            x[batch, channels, :, :] = pywt.idwt2(coeffs, 'haar')

    return x.type(torch.cuda.FloatTensor)

def DFWloss(x,y,loss):
    wtfcA_x,wtfcH_x,wtfcV_x,wtfcD_x = DFWT(x)
    wtfcA_y, wtfcH_y, wtfcV_y, wtfcD_y = DFWT(y)
    loss += torch.nn.L1Loss(wtfcA_x,wtfcA_y)
    loss += torch.nn.L1Loss(wtfcH_x, wtfcH_y)
    loss += torch.nn.L1Loss(wtfcV_x, wtfcV_y)
    loss += torch.nn.L1Loss(wtfcD_x, wtfcD_y)
    return loss




def DFWT(x):
    wtfcA = np.zeros([x.shape[0],x.shape[1],x.shape[2]//2,x.shape[3]//2])
    wtfcH = np.zeros([x.shape[0],x.shape[1],x.shape[2]//2,x.shape[3]//2])
    wtfcV = np.zeros([x.shape[0],x.shape[1],x.shape[2]//2,x.shape[3]//2])
    wtfcD = np.zeros([x.shape[0],x.shape[1],x.shape[2]//2,x.shape[3]//2])
    B = x.shape[0]
    C = x.shape[1]
    for batch in range(B):
        for channels in range(C):
            coeffs = pywt.dwt2(x[batch, channels, :, :].data.cpu().numpy(), 'haar')
            wtfcA[batch, channels, :, :], (wtfcH[batch, channels, :, :], wtfcV[batch, channels, :, :], wtfcD[batch, channels, :, :]) = coeffs
    return torch.from_numpy(wtfcA).type(torch.cuda.FloatTensor), torch.from_numpy(wtfcH).type(torch.cuda.FloatTensor), torch.from_numpy(wtfcV).type(torch.cuda.FloatTensor), torch.from_numpy(wtfcD).type(torch.cuda.FloatTensor)


def iDFWT(wtfcA, wtfcH, wtfcV, wtfcD):
    wtfcA = wtfcA.data.cpu().numpy()
    wtfcH = wtfcH.data.cpu().numpy()
    wtfcV = wtfcV.data.cpu().numpy()
    wtfcD = wtfcD.data.cpu().numpy()
    B = wtfcA.shape[0]
    C = wtfcA.shape[1]
    x = np.zeros([B,C,wtfcA.shape[2]*2,wtfcA.shape[3]*2])
    for batch in range(B):
        for channels in range(C):
            coeffs = wtfcA[batch, channels, :, :],(wtfcH[batch, channels, :, :],wtfcV[batch, channels, :, :],wtfcD[batch, channels, :, :])
            x[batch, channels, :, :] = pywt.idwt2(coeffs, 'haar')
    x = torch.from_numpy(x)
    return x.type(torch.cuda.FloatTensor)

def crop_image(img, d=32):
    """
    Make dimensions divisible by d

    :param pil img:
    :param d:
    :return:
    """

    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)

    bbox = [
        int((img.size[0] - new_size[0]) / 2),
        int((img.size[1] - new_size[1]) / 2),
        int((img.size[0] + new_size[0]) / 2),
        int((img.size[1] + new_size[1]) / 2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped


def crop_np_image(img_np, d=32):
    return torch_to_np(crop_torch_image(np_to_torch(img_np), d))


def crop_torch_image(img, d=32):
    """
    Make dimensions divisible by d
    image is [1, 3, W, H] or [3, W, H]
    :param pil img:
    :param d:
    :return:
    """
    new_size = (img.shape[-2] - img.shape[-2] % d,
                img.shape[-1] - img.shape[-1] % d)
    pad = ((img.shape[-2] - new_size[-2]) // 2, (img.shape[-1] - new_size[-1]) // 2)

    if len(img.shape) == 4:
        return img[:, :, pad[-2]: pad[-2] + new_size[-2], pad[-1]: pad[-1] + new_size[-1]]
    assert len(img.shape) == 3
    return img[:, pad[-2]: pad[-2] + new_size[-2], pad[-1]: pad[-1] + new_size[-1]]


def get_params(opt_over, net, net_input, downsampler=None):
    """
    Returns parameters that we want to optimize over.
    :param opt_over: comma separated list, e.g. "net,input" or "net"
    :param net: network
    :param net_input: torch.Tensor that stores input `z`
    :param downsampler:
    :return:
    """

    opt_over_list = opt_over.split(',')
    params = []

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters()]
        elif opt == 'down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params


def get_image_grid(images_np, nrow=8):
    """
    Creates a grid from a list of images by concatenating them.
    :param images_np:
    :param nrow:
    :return:
    """
    images_torch = [torch.from_numpy(x).type(torch.FloatTensor) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()


def plot_image_grid(name, images_np, interpolation='lanczos', output_path="output/"):
    """
    Draws images in a grid

    Args:
        images_np: list of images, each image is np.array of size 3xHxW or 1xHxW
        nrow: how many images will be in one row
        interpolation: interpolation used in plt.imshow
    """
    assert len(images_np) == 2 
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, 2)

    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)

    plt.savefig(output_path + "{}.png".format(name))


def save_image(name, image_np, output_path="output/"):
    p = np_to_pil(image_np)
    p.save(output_path + "{}.jpg".format(name))

def video_to_images(file_name, name):
    video = prepare_video(file_name)
    for i, f in enumerate(video):
        save_image(name + "_{0:03d}".format(i), f)

def images_to_video(images_dir ,name, gray=True):
    num = len(glob.glob(images_dir +"/*.jpg"))
    c = []
    for i in range(num):
        if gray:
            img = prepare_gray_image(images_dir + "/"+  name +"_{}.jpg".format(i))
        else:
            img = prepare_image(images_dir + "/"+name+"_{}.jpg".format(i))
        print(img.shape)
        c.append(img)
    save_video(name, np.array(c))

def save_heatmap(name, image_np):
    cmap = plt.get_cmap('jet')

    rgba_img = cmap(image_np)
    rgb_img = np.delete(rgba_img, 3, 2)
    save_image(name, rgb_img.transpose(2, 0, 1))


def save_graph(name, graph_list, output_path="output/"):
    plt.clf()
    plt.grid(ls='--')
    axes = plt.gca()
    #axes.set_xlim([xmin, xmax])
    axes.set_ylim([0, 15])
    plt.plot(graph_list,color='blue')
    plt.savefig(output_path + name + ".png")

def save_loss(name, graph_list, output_path="output/"):
    plt.clf()
    plt.grid(ls='--')
    axes = plt.gca()
    #axes.set_xlim([xmin, xmax])
    axes.set_ylim([0, 1])
    plt.plot(graph_list,color='red')
    plt.savefig(output_path + name + ".png")


def create_augmentations(np_image):
    """
    convention: original, left, upside-down, right, rot1, rot2, rot3
    :param np_image:
    :return:
    """
    aug = [np_image.copy(), np.rot90(np_image, 1, (1, 2)).copy(),
           np.rot90(np_image, 2, (1, 2)).copy(), np.rot90(np_image, 3, (1, 2)).copy()]
    flipped = np_image[:,::-1, :].copy()
    aug += [flipped.copy(), np.rot90(flipped, 1, (1, 2)).copy(), np.rot90(flipped, 2, (1, 2)).copy(), np.rot90(flipped, 3, (1, 2)).copy()]
    return aug


def create_video_augmentations(np_video):
    """
        convention: original, left, upside-down, right, rot1, rot2, rot3
        :param np_video:
        :return:
        """
    aug = [np_video.copy(), np.rot90(np_video, 1, (2, 3)).copy(),
           np.rot90(np_video, 2, (2, 3)).copy(), np.rot90(np_video, 3, (2, 3)).copy()]
    flipped = np_video[:, :, ::-1, :].copy()
    aug += [flipped.copy(), np.rot90(flipped, 1, (2, 3)).copy(), np.rot90(flipped, 2, (2, 3)).copy(),
            np.rot90(flipped, 3, (2, 3)).copy()]
    return aug


def save_graphs(name, graph_dict, output_path="output/"):
    """

    :param name:
    :param dict graph_dict: a dict from the name of the list to the list itself.
    :return:
    """
    plt.clf()
    fig, ax = plt.subplots()
    for k, v in graph_dict.items():
        ax.plot(v, label=k)
        # ax.semilogy(v, label=k)
    ax.set_xlabel('iterations')
    # ax.set_ylabel(name)
    ax.set_ylabel('MSE-loss')
    # ax.set_ylabel('PSNR')
    plt.legend()
    plt.savefig(output_path + name + ".png")


def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img


def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size.

    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0] != -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np


def prepare_image(file_name):
    """
    loads makes it divisible
    :param file_name:
    :return: the numpy representation of the image
    """
    img_pil = crop_image(get_image(file_name, -1)[0], d=32)
    return pil_to_np(img_pil)


def prepare_video(file_name, folder="output/"):
    data = skvideo.io.vread(folder + file_name)
    return crop_torch_image(data.transpose(0, 3, 1, 2).astype(np.float32) / 255.)[:35]


def save_video(name, video_np, output_path="output/"):
    outputdata = video_np * 255
    outputdata = outputdata.astype(np.uint8)
    skvideo.io.vwrite(output_path + "{}.mp4".format(name), outputdata.transpose(0, 2, 3, 1))


def prepare_gray_image(file_name):
    img = prepare_image(file_name)
    return np.array([np.mean(img, axis=0)])


def pil_to_np(img_PIL, with_transpose=True):
    """
    Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
        # this is alpha channel
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def fft(img_torch):
    img_np = torch_to_np(img_torch)
    fshift_np = np.fft.fftshift(np.fft.fft2(img_np))
    return np_to_torch(fshift_np).type(torch.cuda.FloatTensor)


def ifft(img_torch):
    fshift= torch_to_np(img_torch)
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)
    return np_to_torch(iimg).type(torch.cuda.FloatTensor)


def median(img_np_list):
    """
    assumes C x W x H [0..1]
    :param img_np_list:
    :return:
    """
    assert len(img_np_list) > 0
    l = len(img_np_list)
    shape = img_np_list[0].shape
    result = np.zeros(shape)
    for c in range(shape[0]):
        for w in range(shape[1]):
            for h in range(shape[2]):
                result[c, w, h] = sorted(i[c, w, h] for i in img_np_list)[l//2]
    return result


def average(img_np_list):
    """
    assumes C x W x H [0..1]
    :param img_np_list:
    :return:
    """
    assert len(img_np_list) > 0
    l = len(img_np_list)
    shape = img_np_list[0].shape
    result = np.zeros(shape)
    for i in img_np_list:
        result += i
    return result / l

def rgb2y_CWH_nol(x):
    return  0.257 * x[0, :, :] + 0.504 * x[1, :, :] + 0.098 * x[2, :, :] + 0.0625

def rgb2y_CWH_nol_torch(x):
    y = 0.299 * x[0,0, :, :] +(-0.169) * x[0,1, :, :] + (0.499) * x[0,2, :, :] + 0.
    cb = 0.587 * x[0, 0, :, :] + (-0.331) * x[0, 1, :, :] + (-0.418) * x[0, 2, :, :] + 0.5
    cr = 0.114 * x[0, 0, :, :] + (0.499) * x[0, 1, :, :] + (-0.0813) * x[0, 2, :, :] + 0.5
    return y.unsqueeze(0).unsqueeze(0),cb.unsqueeze(0).unsqueeze(0),cr.unsqueeze(0).unsqueeze(0)

def rgb2ycbcr(rgb_image):
    """convert rgb into ycbcr"""
    if len(rgb_image.shape)!=3 or rgb_image.shape[2]!=3:
        raise ValueError("input image is not a rgb image")
    rgb_image = rgb_image.astype(np.float32)
    # 1：创建变换矩阵，和偏移量
    transform_matrix = np.array([[0.257, 0.564, 0.098],
                                 [-0.148, -0.291, 0.439],
                                 [0.439, -0.368, -0.071]])
    shift_matrix = np.array([16, 128, 128])
    ycbcr_image = np.zeros(shape=rgb_image.shape)
    w, h, _ = rgb_image.shape
    # 2：遍历每个像素点的三个通道进行变换
    for i in range(w):
        for j in range(h):
            ycbcr_image[i, j, :] = np.dot(transform_matrix, rgb_image[i, j, :]) + shift_matrix
    return ycbcr_image


"""
YUV
[[[[0.299, -0.169, 0.499],
[0.587, -0.331, -0.418],
[0.114, 0.499, -0.0813]]]]

[[[[1., 1., 1.],
[0., -0.34413999, 1.77199996],
[1.40199995, -0.71414, 0.]]]]


"""
def ycbcr2rgb(ycbcr_image):
    """convert ycbcr into rgb"""
    if len(ycbcr_image.shape)!=3 or ycbcr_image.shape[2]!=3:
        raise ValueError("input image is not a rgb image")
    ycbcr_image = ycbcr_image.astype(np.float32)
    transform_matrix = np.array([[0.257, 0.564, 0.098],
                                 [-0.148, -0.291, 0.439],
                                 [0.439, -0.368, -0.071]])
    transform_matrix_inv = np.linalg.inv(transform_matrix)
    shift_matrix = np.array([16, 128, 128])
    rgb_image = np.zeros(shape=ycbcr_image.shape)
    w, h, _ = ycbcr_image.shape
    for i in range(w):
        for j in range(h):
            rgb_image[i, j, :] = np.dot(transform_matrix_inv, ycbcr_image[i, j, :]) - np.dot(transform_matrix_inv, shift_matrix)
    return rgb_image.astype(np.uint8)


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    elif img_np.shape[0] == 3:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)
    else:
        pass

    return Image.fromarray(ar)


def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]
