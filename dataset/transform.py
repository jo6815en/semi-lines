import random

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms


def crop(img, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))

    return img


def hflip(img, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

# Define the color mapping
color_map = {
    (128, 10, 129): 1, #Spruce
    (0, 0, 0): 0, #Background
    (65, 20, 19): 0, #Obsacle
    (144, 96, 0): 1, #Tree
    (147, 228, 202): 0, #Lake
    (57, 240, 237): 0, #Track
    (64, 0, 160): 0, #Ground
    (130, 44, 136): 1, #Birch
    (104, 173, 142): 1, #Pine
}

def map_to_nine(img):

    # Create a single channel image with the same height and width as the RGB image
    single_channel_img = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)

    for rgb, target in color_map.items():
        single_channel_img[(img == rgb).all(-1)] = target

    return single_channel_img

def normalize(img, mask=None):
    #img = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #])(img)

    img = transforms.Compose([
        transforms.ToTensor(),
    ])(img)

    if mask is not None:
        mask = np.array(mask)
        if len(mask.shape) == 3 and mask.shape[2] > 1:
            mask = map_to_nine(mask)
        mask = torch.from_numpy(mask).long()
        return img, mask
    return img


def resize(img, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    return img


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_split(img_size, p=0.5, ratio_1=0.3, ratio_2=0.7, p2=0.5):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    cutmix = int(np.random.uniform(ratio_1, ratio_2) * img_size)
    if random.random() > p2:
        mask[:, cutmix:] = 1
    else:
        mask[cutmix:, :] = 1

    return mask



def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask
