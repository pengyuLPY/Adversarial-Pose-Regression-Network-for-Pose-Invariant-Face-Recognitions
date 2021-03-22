import sys
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
import torch.distributed as dist


from train_test import *
from util.imageLoader import ImageLabelBinFolder
import util.data_transformer as transforms
import util.target_transformer as target_transformer
import util.logger as logger 
from util.lr_policy import *

import util.data_landmark_transformer as transforms
import util.target_transformer as target_transformer
import util.data_transformer as data_transforms

from config import *

if __name__ == '__main__':
    print ('Connecting...')
    if distributed:
        dist.init_process_group(
                                backend=dist_backend, 
                                init_method=dist_url,
                                world_size=worlds_size,
                                 rank=rank
                                )
    print ('Connecting complete')
    sys.stdout = logger.Logger(snapshot_prefix+'_'+str(time.time())+'.log')
    print ('Data loading...')
    normalize = data_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    train_dataset = ImageLabelBinFolder(
        root=trainRoot, proto=trainProto, binRoot=trainRoot, bin_len=102, sign_replace=False,
        transform=transforms.Compose([
            transforms.Scale(resize_shape),
            transforms.RandomCrop((crop_shape)),
        ]),
        data_transform=data_transforms.Compose([
            data_transforms.ToTensor(),
            normalize,
        ]),
        target_transform=target_transformer.ToInt()
       )

    test_dataset = ImageLabelBinFolder(
        root=testRoot, proto=testProto, binRoot=testRoot, bin_len=102, sign_replace=False,
        transform=transforms.Compose([
            transforms.Scale((crop_shape)),
        ]),
        data_transform=data_transforms.Compose([
            data_transforms.Scale((crop_shape)),
            data_transforms.ToTensor(),
            normalize,
        ]),
        target_transform=target_transformer.ToInt()
       )
        
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batchSize, shuffle=(train_shuffle and train_sampler is None),
        num_workers = workers, pin_memory=True, sampler = train_sampler, drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batchSize, shuffle=test_shuffle,
        num_workers = workers, pin_memory=True
    )


    print ('Data loading complete')
#################################  MODEL INIT ##################
    print ('Model Initing...')
    best_prec1 = 0
    cudnn.benchmark = cudnn_use

    if distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()
    criterion_l1 = torch.nn.L1Loss().cuda()
    lr_obj = lr_class()
    #optimizer = torch.optim.SGD(model.parameters(), lr_obj.base_lr,
    #                            momentum=momentum,
    #                            weight_decay=weight_decay)
    optimizer = torch.optim.SGD([
                                {'params': model.module.feature_extraction_net.parameters()},
                                {'params':model.module.fc.parameters()}
                                ],
                                lr_obj.base_lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    optimizer_discriminator = torch.optim.SGD(model.module.discriminator.parameters(), 
                                lr_obj.base_lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    if resume:
        if os.path.isfile(resume_file):
            print("checkpoint => loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            IterationNum[0] = checkpoint['iteration'] 
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            print("checkpoint => loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("checkpoint => no checkpoint found at '{}'".format(resume))
            exit()
    if pretrained:
            if os.path.isfile(pretrained_file):
                    print("pretrain => loading checkpoint '{}'".format(pretrained_file))
                    checkpoint = torch.load(pretrained_file)
                    model.load_state_dict(checkpoint['state_dict'], False)
            else:
                    print("pretrain => no checkpoint found at '{}'".format(pretrained))
                    exit()

    print ('Model Initing complete')

#################################  TRAINING  ##################
    if test_init:
        print ('Testing...')
        prec1 = test(test_loader, model, criterion, criterion_l1, epoch=0, test_iter=test_iter)

    epoch = -1
    lr_obj = lr_class()
    for epoch in range(start_epoch, max_epoch+1):
        print ('Training... epoch:',epoch)
        adjust_learning_rate(lr_obj, optimizer, epoch)
        adjust_learning_rate(lr_obj, optimizer_discriminator, epoch)
        if distributed:
            train_sampler.set_epoch(epoch)

        train(train_loader, model, criterion, criterion_l1, optimizer, optimizer_discriminator, lr_obj, epoch, train_iter_per_epoch, display, IterationNum, snapshot, snapshot_prefix)
        print ('Testing... epoch:',epoch)
        prec1 = test(test_loader, model, criterion, criterion_l1, epoch=epoch, test_iter=test_iter)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(
            {
                'epoch': epoch+1,
                'iteration': IterationNum[0],
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
                'optimizer_discriminator': optimizer_discriminator.state_dict(),
            }, is_best, snapshot_prefix+'_epoch_'+str(epoch)
        )
        

            
    print ('Training complete')

