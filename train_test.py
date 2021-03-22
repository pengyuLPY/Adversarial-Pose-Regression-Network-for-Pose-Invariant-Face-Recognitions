import os
import shutil
import time
import numpy as np
import torch
#import torch.nn as nn
#import torch.nn.parallel
#import torch.backends.cudnn as cudnn
#import torch.distributed as dist
#import torch.optim
#import torch.utils.data
#import torch.utils.data.distributed
from tqdm import tqdm
from config import mean_landmark,frequency_D,frequency_G,loss_weight

def train(train_loader, model, criterion_cross_entropy, criterion_l1, optimizer, optimizer_discriminator, lr_obj, epoch, train_iter_per_epoch, display, IterationNum, snapshot, snapshot_prefix, gpu=None):

    model.train()


    print('Training length is {}'.format(len(train_loader)))
    i = -1
    end = time.time()
    for data, target, target_landmark in train_loader:
        i += 1

        if gpu is None:
            target = target.cuda()
            data = data.cuda()
            target_landmark = target_landmark.cuda()
            mean_landmark_matrix = torch.from_numpy(mean_landmark).cuda()
            ones_vec = torch.ones(target_landmark.size(0),1).cuda()
            mean_landmark_matrix = torch.mm(ones_vec, mean_landmark_matrix)
        else:
            target = target.cuda(gpu, non_blocking=True)
            data = data.cuda(gpu, non_blocking=True)
            target_landmark = target_landmark.cuda(gpu, non_blocking=True)
            mean_landmark_matrix = torch.from_numpy(mean_landmark).cuda(gpu, non_blocking=True)
            ones_vec = torch.ones(target_landmark.size(0),1).cuda(gpu, non_blocking=True)
            mean_landmark_matrix = torch.mm(ones_vec, mean_landmark_matrix)

        output,pred_landmark = model(data, target, testing=False)

        loss_cross_entropy = criterion_cross_entropy(output,target)
        loss_D = criterion_l1(pred_landmark, target_landmark)
        loss_G = criterion_l1(pred_landmark, mean_landmark_matrix)
        loss_T = criterion_l1(target_landmark,mean_landmark_matrix)

        # D update
        if IterationNum[0] % frequency_D == 0:
            optimizer_discriminator.zero_grad()
            loss_D = loss_D * loss_weight
            loss_D.backward(retain_graph=True)
        # R update
        if IterationNum[0] % frequency_G == 0:
            loss = loss_cross_entropy + loss_G * loss_weight
        else:
            loss = loss_cross_entropy
        optimizer.zero_grad()
        loss.backward()

        if IterationNum[0] % frequency_D == 0:
            optimizer_discriminator.step()
        optimizer.step()
        


        loss_cross_entropy_show = loss_cross_entropy.item()
        loss_D_show = loss_D.item()
        loss_G_show = loss_G.item()
        loss_T_show = loss_T.item()

        run_time = time.time() - end
        end = time.time()
        IterationNum[0] += 1

        if IterationNum[0] % display == 0:
            print('Device_ID:' + str(torch.cuda.current_device()) + ', '
                  'Epoch:' + str(epoch) + ', ' + 'Iteration:' + str(IterationNum[0]) + ', ' +
                  'TrainSpeed = '+ str(run_time) + ', ' +
                  'lr = ' + str(optimizer.param_groups[0]['lr']) +', ' +
                  'Trainloss_D = '+str(loss_D_show)+', '+
                  'Trainloss_G = '+str(loss_G_show)+', '+
                  'Trainloss_T = '+str(loss_T_show)+', '+
                  'Trainloss_crossEntropy = ' + str(loss_cross_entropy_show))

        if IterationNum[0] % snapshot == 0:
                save_checkpoint(
                    {
                        'epoch': epoch+1,
                        'state_dict': model.state_dict(),
                        'best_pre1': -1,
                        'optimizer' : optimizer.state_dict(),
                        'optimizer_discriminator':optimizer_discriminator.state_dict(),
                    }, False, snapshot_prefix+'_iter_'+str(IterationNum[0])
                )
        if train_iter_per_epoch!=0 and IterationNum[0] % train_iter_per_epoch == 0:
                break;        

def test(test_loader, model, criterion_cross_entropy, criterion_l1, epoch, test_iter = 0, gpu=None):
    model.eval()


    end = time.time()
    loss_cross_entropy_show = 0
    loss_D_show = 0
    loss_G_show = 0
    prec1 = 0
    num = 0.0
    i = -1
    print('Testing length is {}'.format(len(test_loader)))
    with torch.no_grad():
        for data,target,target_landmark in tqdm(test_loader):
            i += 1
            if test_iter != 0 and i == test_iter:
                break

            if gpu is None:
                target = target.cuda()
                data = data.cuda()
                target_landmark = target_landmark.cuda()
                mean_landmark_matrix = torch.from_numpy(mean_landmark).cuda()
                ones_vec = torch.ones(target_landmark.size(0),1).cuda()
                mean_landmark_matrix = torch.mm(ones_vec, mean_landmark_matrix)
            else:
                target = target.cuda(gpu, non_blocking=True)
                data = data.cuda(gpu, non_blocking=True)
                target_landmark = target_landmark.cuda(gpu, non_blocking=True)
                mean_landmark_matrix = torch.from_numpy(mean_landmark).cuda(gpu, non_blocking=True)
                ones_vec = torch.ones(target_landmark.size(0),1).cuda(gpu, non_blocking=True)
                mean_landmark_matrix = torch.mm(ones_vec, mean_landmark_matrix)

            (output,output1),pred_landmark = model(data, target, testing=True)
            loss_cross_entropy = criterion_cross_entropy(output, target)
            loss_cross_entropy_show += loss_cross_entropy.item()
            loss_D = criterion_l1(pred_landmark, target_landmark)
            loss_D_show += loss_D.item()
            loss_G = criterion_l1(pred_landmark, mean_landmark_matrix)
            loss_G_show += loss_G.item()

            prec1_t = accuracy(output1.data, target)
            prec1 += prec1_t[0][0]
            num += 1


        run_time = time.time() - end
        if test_iter != 0:
            print('Device_ID:' + str(torch.cuda.current_device()) + ', '
                  'Epoch:' + str(epoch) + ', ' +
                  'TestSpeed = ' + str(run_time) + ', ' +
                  'Testloss_crossEntropy = '+str(loss_cross_entropy_show/num)+', '+
                  'Testloss_D = '+str(loss_D_show/num)+', '+
                  'Testloss_G = '+str(loss_G_show/num)+', '+
                  'TestAccuracy = '+str(prec1/num))
        else:
            print('Device_ID:' + str(torch.cuda.current_device()) + ', '
                 'Epoch:' + str(epoch) + ', ' +
                 'test_iter:' + str(test_iter) + ', ' +
                  'TestSpeed = ' + str(run_time) + ', ' +
                 'Testloss_crossEntropy = ' + str(loss_cross_entropy_show / num) + ', ' +
                  'Testloss_D = '+str(loss_D_show/num)+', '+
                  'Testloss_G = '+str(loss_G_show/num)+', '+
                 'TestAccuracy = ' + str(prec1 / num))
    return prec1 / num

def extract_feature_per_img(test_loader, model, root):
        model.eval()
        pre = 0

        for i,(data,target,target_landmark,imglist) in enumerate(test_loader):
                if i % 100 == 0:
                        print (i,'vs',test_loader.__len__())
                data = data.cuda()
                output = model(data, extract_feature=True)

                output_data = output.data.cpu()
                output_np = output_data.numpy()
                output_np = output_np.astype(np.float32)
                for i,img in enumerate(imglist):
                    saveName = root+img+'.bin'
                    path,name = os.path.split(saveName)
                    #print saveName
                    #exit()
                    if os.path.exists(path) == False:
                        command = 'mkdir -p ' + path
                        os.system(command)
                    output_np[i,:].tofile(saveName)



def extract_feature(test_loader, model):
        result = None
        model.eval()
        pre = 0
        for i,(data,target,target_landmark) in enumerate(test_loader):
                if i % 100 == 0:
                        print (i,'vs',test_loader.__len__())
                target = target.cuda()
                data = data.cuda()
                output = model(data, extract_feature=True)

                if result is None:
                        size = np.array(output.data.cpu().size())
                        n = size[0]
                        size[0] = test_loader.dataset.__len__()
                        result = torch.FloatTensor(*size).zero_()

                result[pre:pre+n,:] = output.data.cpu().clone()
                pre = pre+n

        return result
        

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res





def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename += '.pytorch'
    print ('saving snapshot in',filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
    print ('saving complete')
