from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps, ImageFilter

try:
    import accimage
except ImportError:
    accimage = None
import scipy.ndimage as ndimage
import numpy as np
import numbers
import types
import collections

import cv2

def updateScaleLabel(label, ih, iw, oh, ow):
    label[:,0] = label[:,0]*ow/iw
    label[:,1] = label[:,1]*oh/ih


    return label

def updateCropLabel(label, bx, by):
    label[:,0] -= bx
    label[:,1] -= by

    return label

def updateFlipLabel(label, w):
    label[:,0] = w-label[:,0]
    return label


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, labels, prev_img_index=-1):
        labels = labels.reshape(-1,2)
        prev_img = img.copy()
        for i,t in enumerate(self.transforms):
            if i == prev_img_index:
                prev_img = img.copy()
            img,labels,img_size = t(img, labels)
        return img,prev_img,labels.reshape(-1),img_size



class Scale(object):
    """Rescale the input PIL.Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR, ds=0):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
        self.ds = ds

    def __call__(self, img, label):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """



        if isinstance(self.size, int):
            w, h = img.size

            #if (w <= h and w == self.size) or (h <= w and h == self.size):
            #    return img, label
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)

                if self.ds != 0:
                    oh = int(oh/self.ds) * self.ds

                return img.resize((ow, oh), self.interpolation), updateScaleLabel(label=label, ih=h, iw=w, oh=oh, ow=ow), np.array([oh, ow])
            else:
                oh = self.size
                ow = int(self.size * w / h)
                if self.ds != 0:
                    ow = int(ow/self.ds) * self.ds

                return img.resize((ow, oh), self.interpolation), updateScaleLabel(label=label, ih=h, iw=w, oh=oh, ow=ow), np.array([oh, ow])
        else:
            w, h = img.size

            oh = self.size[0]
            ow = self.size[1]

            return img.resize((ow, oh), self.interpolation), updateScaleLabel(label=label, ih=h, iw=w, oh=oh, ow=ow), np.array([oh, ow])




class RandomScale(object):
    """Rescale the input PIL.Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, ratio, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, img, label):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """



        if isinstance(self.size, int):
            w, h = img.size
            ratio_cur = random.uniform(1.0/self.ratio, self.ratio)

            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img, label
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                ow = int(ow*ratio_cur)
                oh = int(oh*ratio_cur)

                return img.resize((ow, oh), self.interpolation), updateScaleLabel(label=label, ih=h, iw=w, oh=oh, ow=ow), np.array([oh, ow])
            else:
                oh = self.size
                ow = int(self.size * w / h)
                ow = int(ow*ratio_cur)
                oh = int(oh*ratio_cur)



                return img.resize((ow, oh), self.interpolation), updateScaleLabel(label=label, ih=h, iw=w, oh=oh, ow=ow), np.array([oh, ow])
        else:
            w, h = img.size

            ratio_cur = random.uniform(1.0/self.ratio, self.ratio)
            oh = self.size[0]
            ow = self.size[1]
            ow = int(ow*ratio_cur)
            oh = int(oh*ratio_cur)

            return img.resize((ow, oh), self.interpolation), updateScaleLabel(label=label, ih=h, iw=w, oh=oh, ow=ow), np.array([oh, ow])

class CenterCrop(object):
    """Crops the given PIL.Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, label):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), updateCropLabel(label=label, bx=x1, by=y1), np.array([th, tw])


class RandomCrop(object):
    """Crop the given PIL.Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, label):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, label
        if h > w:
           tw, th = self.size

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)),updateCropLabel(label=label, bx=x1, by=y1), np.array([th, tw])




class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, img, label):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        img_w, img_h = img.size
        if random.random() < self.flip_prob:
            return img.transpose(Image.FLIP_LEFT_RIGHT), updateFlipLabel(label, w=img.size[0]), np.array([img_h, img_w])
        return img, label, np.array([img_h, img_w])


class RandomSizedCrop(object):
    """Crop the given PIL.Image to random size and aspect ratio.

    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, size_ratio=2.0, aspect_ratio=4.0/3.0, interpolation=Image.BILINEAR):
        self.size = size
        self.size_ratio = size_ratio
        self.aspect_raio = aspect_ratio
        assert size_ratio >= 1
        assert aspect_ratio >= 1
        self.interpolation = interpolation

    def __call__(self, img, label):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(1/self.size_ratio, self.size_ratio) * area
            aspect_ratio = random.uniform(1/self.aspect_raio, self.aspect_raio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                new_label = updateCropLabel(label=label, bx=x1, by=y1)
                assert (img.size == (w, h))

                return img.resize((self.size[1], self.size[0]), self.interpolation), updateScaleLabel(label=new_label, ih=h, iw=w, oh=self.size[0], ow=self.size[1]), np.array([self.size[0], self.size[1]])

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        img,label = scale(img,label)
        img,label = crop(img, label)

        return img, label


class RandomScaleFixOutput(object):
    def __init__(self, fixSize, bbSize_min, bbSize_max, ratio_min = None, ratio_max = None, fixId=None, interpolation=Image.BILINEAR):
        self.fixSize = fixSize
        self.bbSize_min = bbSize_min
        self.bbSize_max = bbSize_max
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.fixId = fixId
        self.interpolation = interpolation

    def __call__(self, img, label):
        w,h = img.size

        fh,fw = self.fixSize

        if self.fixId is None:
            ind = int(random.uniform(0, label.shape[0]))
        else:
            ind = self.fixId
        w_bb = label[ind,2] - label[ind,0]
        h_bb = label[ind,3] - label[ind,1] 
        
        randomSize = random.uniform(self.bbSize_min, self.bbSize_max)
        randomRatio = randomSize/min(w_bb, h_bb)  
        if self.ratio_min is not None:
            randomRatio = max(randomRatio, self.ratio_min)
        if self.ratio_max is not None:
            randomRatio = min(randomRatio, self.ratio_max) 
        
        ow = int(w * randomRatio)
        oh = int(h * randomRatio)
        
        img = img.resize((ow,oh), self.interpolation)
        label = updateScaleLabel(label=label, ih=h, iw=w, oh=oh, ow=ow)
        
        bias_x =  label[ind,0] - fw/2 
        bias_y =  label[ind,1] - fh/2
        bias_x = max(0, min(bias_x, ow-fw))
        bias_y = max(0, min(bias_y, oh-fh))
        
        img = img.crop((bias_x, bias_y, bias_x + fw, bias_y + fh))
        label = updateCropLabel(label=label, bx=bias_x, by=bias_y)
        
        
        img_w = min(ow, fw)
        img_h = min(oh, fh)
        
        return img, label, np.array([img_h, img_w]) 



class CenterCropWithOffset(object):
    """Crops the given PIL.Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, x_size, y_size, x_offset, y_offset, x_joggle, y_joggle, ignore_fault=False):
        self.x_size = x_size
        self.y_size = y_size
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.x_joggle = x_joggle
        self.y_joggle = y_joggle
        self.ignore_fault = ignore_fault


    def __call__(self, img, label):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th = self.y_size
        tw = self.x_size

        x1 = int(round((w - tw) / 2.)) + self.x_offset + random.uniform(0,self.x_joggle)*(random.uniform(0,1)-0.5)*2
        y1 = int(round((h - th) / 2.)) + self.y_offset + random.uniform(0,self.y_joggle)*(random.uniform(0,1)-0.5)*2

        x1 = max(0,x1)
        x2 = min(x1+tw, w)
        y1 = max(0,y1)
        y2 = min(y1+th, h)
        if y2-y1 != self.y_size:
            if self.ignore_fault:
                y1 = 0
                th = y2;
            else:
                raise(RuntimeError('(data_transformer)Size Error:y2-y1 != self.y_size:' +
                                   str(y2 - y1) + 'vs' + str(self.y_size)))
        if x2-x1 != self.x_size:
            if self.ignore_fault:
                x1 = 0
                tw = x2
            else:
                raise (RuntimeError('(data_transformer)Size Error:x2-x1 != self.x_size:' +
                                str(x2 - x1) + 'vs' + str(self.x_size)))

        return img.crop((x1, y1, x1 + tw, y1 + th)), updateCropLabel(label, x1, y1), np.array([th, tw])





