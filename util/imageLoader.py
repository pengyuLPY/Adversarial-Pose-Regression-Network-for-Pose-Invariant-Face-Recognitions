import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import torch
from tqdm import tqdm


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    #from torchvision import get_image_backend
    #if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    #else:
    return pil_loader(path)

def opencv_loader(path):
    import cv2
    return cv2.imread(path)





class ImageLabelFolder(data.Dataset):
    class imageLabel:
        def __init__(self, image_path, label_len):
            self.image = image_path
            self.labels = None
            self.label_len = label_len
            if os.path.exists(image_path):
                self.success = True
            else:
                self.success = False

    def __init__(self, root, proto, transform=None, target_transform=None, label_len=1, sign_imglist=False,loader=default_loader, key_index=0, ignore_fault=False):
        protoFile = open(proto,  encoding="utf8", errors='ignore')
        content = protoFile.readlines()
        self.imageLabel_list = []
        self.label_len = label_len
        self.loader = loader
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.sign_imglist = sign_imglist
        self.ignore_fault=ignore_fault
        errorwriter=open('./errorImage.txt','a')

        for line in tqdm(content):
            line = line.strip()
            line_list = line.split(' ')
            imagePath = ''
            for i in range(len(line_list)):
                if i < len(line_list) - self.label_len:
                    imagePath += line_list[i] + ' '
            imagePath = imagePath[:-1]
            imagePath = self.root + imagePath 
            imagePath = os.path.normpath(imagePath)

            cur_imageLabel = self.imageLabel(imagePath,line_list.__len__()-self.label_len)
            label_list = []
            for i in range(line_list.__len__()-self.label_len,line_list.__len__()):
                label_list.append(float(line_list[i]))

            cur_imageLabel.labels = torch.FloatTensor(label_list)
            if cur_imageLabel.success:
                self.imageLabel_list.append(cur_imageLabel)
            else:
                errorwriter.write(line+'\n')

        self.labelgenerator(key_index)
        errorwriter.close()
        print('data size is ',self.imageLabel_list.__len__(), content.__len__())

    def labelgenerator(self, key_index=0):
        self.labels = torch.FloatTensor((self.imageLabel_list.__len__()))
        for i, imglabel in enumerate(self.imageLabel_list):
            self.labels[i] = imglabel.labels[key_index]



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """


        cur_imageLabel = self.imageLabel_list[index]
        if self.ignore_fault:
            try:
                img = self.loader(cur_imageLabel.image)
            except:
                img = Image.new("RGB",(300,300),(0,0,0))
        else:
            img = self.loader(cur_imageLabel.image)
        labels = cur_imageLabel.labels
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            labels = self.target_transform(labels)
        
        if self.sign_imglist:
            return img, labels, cur_imageLabel.image
        else:
            return img, labels

    def __len__(self):
        return len(self.imageLabel_list)







class ImageLabelBinFolder(data.Dataset):
    class imageLabelBin:
        def __init__(self, image_path, label_len, bin, bin_len):
            self.image = image_path
            self.labels = None
            self.label_len = label_len
            self.bin = bin.copy()
            self.bin_len = bin_len
            if self.bin.shape[0] != self.bin_len:
                print ('int imageloader.py bin.shape[0] != cur_imageLabelBin.bin_len',bin.shape[0],cur_imageLabelBin.bin_len)
                exit()

    def __init__(self, root, proto, binRoot, transform=None, data_transform=None, target_transform=None, replace_src='', replace_des='', bin_len=1, miss_default=None, sign_imglist=False,loader=default_loader,key_index=0,label_len=1, sign_replace=True, prev_img=-1):
        protoFile = open(proto,  encoding="utf8", errors='ignore')
        content = protoFile.readlines()
        self.imageLabelBin_list = []
        self.label_len = label_len
        self.loader = loader
        self.root = root
        self.binRoot = binRoot
        self.transform = transform
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.sign_imglist = sign_imglist
        self.miss_default = miss_default
        self.sign_replace = sign_replace
        
        self.prev_img = prev_img

        errorwriter=open('./errorImage.txt','a')
        for line in tqdm(content):
            line = line.strip()

            if sign_replace == False:
                bin_path = self.binRoot + line.split(' #BINSPLIT# ')[1]
                line = line.split(' #BINSPLIT# ')[0]

            line_list = line.split(' ')

            imagePath = ''
            for i in range(len(line_list)):
                if i < len(line_list) - self.label_len:
                    imagePath += line_list[i] + ' '
            imagePath = imagePath[:-1]
            img_path = self.root+imagePath
            img_path = os.path.normpath(img_path)

            if sign_replace:
                bin_path = self.binRoot+imagePath.replace(replace_src, replace_des) + '.bin'
            bin_path = os.path.normpath(bin_path)


            if os.path.isfile(img_path) == False:
                errorwriter.write('img_path: '+img_path+' not exists'+'\n')
                continue
            if os.path.isfile(bin_path) == False: 
                errorwriter.write('bin_path: '+bin_path+' not exists'+'\n')
                if self.miss_default is None:
                    continue
                else:
                    bin_path = self.miss_default
            bin = np.fromfile(bin_path, np.float32)
        
            cur_imageLabelBin = self.imageLabelBin(img_path, line_list.__len__()-1, bin, bin_len)
            label_list = []
            for i in range(line_list.__len__()-self.label_len,line_list.__len__()):
                label_list.append(float(line_list[i]))
        
            cur_imageLabelBin.labels = torch.FloatTensor(label_list)
            self.imageLabelBin_list.append(cur_imageLabelBin)
        
        
        self.labelgenerator(key_index)
        errorwriter.close()
        print ('data size is,',self.imageLabelBin_list.__len__(), content.__len__())

    def labelgenerator(self, key_index=0):
        self.labels = torch.FloatTensor((self.imageLabelBin_list.__len__()))
        for i, imglabel in enumerate(self.imageLabelBin_list):
            self.labels[i] = imglabel.labels[key_index]    

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """


        cur_imageLabelBin = self.imageLabelBin_list[index]

        img = self.loader(cur_imageLabelBin.image)
        labels = cur_imageLabelBin.labels
        bin = cur_imageLabelBin.bin #np.fromfile(cur_imageLabelBin.bin,np.float32)

        prev_img = img
        if self.transform is not None:
            img,prev_img,bin,_ = self.transform(img, bin, self.prev_img)
        if self.target_transform is not None:
            labels = self.target_transform(labels)
        if self.data_transform is not None:
            img = self.data_transform(img)
        if self.data_transform is not None:
            prev_img = self.data_transform(prev_img)

        bin = torch.from_numpy(bin)
        
        if self.sign_imglist:
            return img, labels, bin, cur_imageLabelBin.image
        else:
            if self.prev_img >= 0:
                return img, prev_img, labels, bin
            else:
                 return img, labels, bin

    def __len__(self):
        return len(self.imageLabelBin_list)









