import sys
import os
from train_test import *
from util.imageLoader import ImageLabelBinFolder
import util.data_transformer as transforms
import util.target_transformer as target_transformer
import CNNs.resnet_identityMapping as resnet
import numpy as np
import torch.backends.cudnn as cudnn

import util.data_transformer as data_transforms
import util.data_landmark_transformer as transforms
import util.target_transformer as target_transformer

def extract_feature_main(testRoot, testProto, test_batchSize, pretrained_file, model):
#################################  DATA LOAD  ##################
    print (testProto, 'Data loading...')
    normalize = data_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    test_dataset = ImageLabelBinFolder(
        root=testRoot, proto=testProto, binRoot=testRoot, bin_len=102, sign_replace=False,
        miss_default='mean_landmark.bin',
        transform=transforms.Compose([
            transforms.Scale((224,224)),
        ]),
        data_transform=data_transforms.Compose([
            data_transforms.ToTensor(),
            normalize,
        ]),
        target_transform=target_transformer.ToInt()
       )

    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batchSize, shuffle=False,
        num_workers = 1, pin_memory=True
    )


    print ('Data loading complete')

    #################################  TRAINING  ##################

    print ('Feature Extracting...')
    feature = extract_feature(test_loader, model)
    print ('Feature Extracting complete', feature.size())
    return feature

def SaveToBin(feature, save):
    feature_np = feature.numpy().astype(np.float32)
    path,name = os.path.split(save)
    if os.path.exists(path) == False:
        command = 'mkdir -p ' + path
        os.system(command)    
    feature_np.tofile(save)
    print (feature_np.shape )

if __name__ == '__main__':
    cudnn.benchmark = True

    test_BatchSize = 100
    #################################  MODEL INIT ##################
    print ('Model Initing...')
    model = resnet.resnet50(feature_len=256,num_classes=74974)
    model = torch.nn.DataParallel(model).cuda()
    pretrained_file = r'./snapshot/resnet101_epoch_5.pytorch'
    if os.path.isfile(pretrained_file):
            print("pretrain => loading checkpoint '{}'".format(pretrained_file))
            checkpoint = torch.load(pretrained_file)
            parameters = checkpoint['state_dict']
            model.load_state_dict(parameters)
            print("pretrain => loaded checkpoint '{}' (epoch {})".format(True, checkpoint['epoch']))
    else:
            print("pretrain => no checkpoint found at '{}'".format(pretrained_file))
            exit()

    print ('Model Initing complete')

    testRoot_list = [
                '/home/pengyu.lpy/dataset/verify/lfw/',
                '/home/pengyu.lpy/dataset/verify/IJB/IJB-A/',
           ]
    testProto_list = [
                '/home/pengyu.lpy/dataset/verify/lfw/lfw_align_bin_list.txt',
                '/home/pengyu.lpy/dataset/verify/IJB/IJB-A/img_bin_list.txt',
           ]

    save_list = [
                './bin/lfw_resnet50_arcFace_epoch_16.bin',
                './bin/ijba_resnet50_arcFace_epoch_16.bin',
    ]
    

    if testRoot_list.__len__() != testProto_list.__len__():
        print ('testRoot_list.__len__() != testProto_list.__len__()',testRoot_list.__len__() ,'vs', testProto_list.__len__())
        exit();
    if testRoot_list.__len__() != save_list.__len__():
        print ('testRoot_list.__len__() != save_list.__len__()',testRoot_list.__len__() ,'vs', save_list.__len__())
        exit();    

    for ind in range(testRoot_list.__len__()):
        testRoot = testRoot_list[ind]
        testProto = testProto_list[ind]
        save = save_list[ind]
        feature = extract_feature_main(testRoot, testProto, test_BatchSize, pretrained_file, model)
        print (testProto,':', feature.size())
        SaveToBin(feature, save)
        print ('------------------------------' )

print ('complete')



