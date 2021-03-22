import numpy as np
import CNNs.resnet_identityMapping as resnet
#import CNNs.casia_net as casia_net

worlds_size = 1
distributed = worlds_size > 1
workers = 16
cudnn_use = True
dist_url='tcp://gpu1051.nt12:23456'
dist_backend='nccl'
rank=0

# The Resnet101 Training with MS-Celeb1M
model = resnet.resnet101(feature_len=256,num_classes=74974)
resize_shape = (225,225)
crop_shape = (224,224)
snapshot_prefix = r'./snapshot/resnet101'

# The CASIA-Net Training with CASIA-WebFace
#model = casia_net.CasiaNet(feature_len=320,num_classes=74974)
#resize_shape = (100,100)
#crop_shape = (96,96)
#snapshot_prefix = r'./snapshot/casia_net'

snapshot=100000000

test_init=True
display = 100

# APRN setting
frequency_D = 2 #1000000
frequency_G = 2
mean_landmark = np.fromfile('mean_landmark.bin',np.float32).reshape(1,-1)
loss_weight = 0.2

# Training setting
max_epoch=20
start_epoch = 0
IterationNum = [0]
test_iter = 0
train_iter_per_epoch = 0
resume = False
resume_file = r''
pretrained = False
pretrained_file = r''


trainRoot = r'./Cleaned_MSCeleb1M_Norm/'
trainProto = r'./Cleaned_MSCeleb1M_Norm/meta/recognition_meta_train_bin.txt'
#format: "ImgPath_IDLabel_ #BINSPLIT#_LandmarkBinPath",
#expamle "Img_align_180_220/m.05bnbs/74-FaceId-0.jpg 0 #BINSPLIT# 180_220_landmark/m.05bnbs/74-FaceId-0.jpg.bin\n"

train_shuffle = True
train_batchSize = 120

testRoot = r'./Cleaned_MSCeleb1M_Norm/'
testProto = r'./Cleaned_MSCeleb1M_Norm/meta/recognition_meta_test_bin.txt'
test_shuffle = False
test_batchSize = 100



class lr_class:
    def __init__(self):
        self.base_lr = 0.001
        self.gamma = 0.1
        self.lr_policy = "multistep"
        self.steps = [16,18,19] #[4800,5400,5800]

momentum = 0.9
weight_decay = 0.0005
