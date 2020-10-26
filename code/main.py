import os
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchnet as tnt
from skimage import io
import tools
import custom
from torch.utils.data import DataLoader
import cv2
import network
import shutil

train_areas = np.load('/home/mariapap/DATA/SPACENET7/EXPS/Fsplit/Ftrain.npy').tolist()
val_areas = np.load('/home/mariapap/DATA/SPACENET7/EXPS/Fsplit/Fval.npy').tolist()

csv_file_train = '/home/mariapap/DATA/SPACENET7/EXPS/xys/myxys_train.csv'
csv_file_val = '/home/mariapap/DATA/SPACENET7/EXPS/xys/myxys_val.csv'

patch_size=32
nb_dates=19

change_dataset =  custom.MyDataset(csv_file_train, train_areas, patch_size, nb_dates)
mydataset = DataLoader(change_dataset, batch_size=2, shuffle=True, drop_last=True)
change_dataset_val = custom.MyDataset(csv_file_val, val_areas, patch_size, nb_dates)
mydataset_val = DataLoader(change_dataset_val, batch_size=1, shuffle=False, drop_last=True)

model = tools.to_cuda(network.U_Net(4,2,32)) #here 4 is the number of input channels, 2 is the number of output categories (change or no change)
                                             # and 32 is the employed patch size

base_lr=0.0001
optimizer = optim.Adam(model.parameters(), lr=base_lr)
weight_tensor=torch.FloatTensor(2)
weight_tensor[0]= 0.05
weight_tensor[1]= 0.95
criterion_ch=tools.to_cuda(nn.CrossEntropyLoss(tools.to_cuda(weight_tensor)))

build_tensor=torch.FloatTensor(2)
build_tensor[0]= 0.1
build_tensor[1]= 0.9
criterion_segm=tools.to_cuda(nn.CrossEntropyLoss(tools.to_cuda(build_tensor)))

diff_tensor=torch.FloatTensor(2)
diff_tensor[0]= 0.04
diff_tensor[1]= 0.96
criterion_diff=tools.to_cuda(nn.CrossEntropyLoss(tools.to_cuda(diff_tensor)))

confusion_matrix = tnt.meter.ConfusionMeter(2, normalized=True)
epochs=30

save_folder = 'models' #where to save the models and training progress
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.mkdir(save_folder)

ff=open('./' + save_folder + '/progress.txt','w')

for epoch in range(1, epochs+1):
    model.train()
    train_losses = []
    confusion_matrix.reset()
    iter_ = 0

    for i, batch, in enumerate(tqdm(mydataset)):
        img_batch, lbl_batch, bld_batch = batch
        img_batch, lbl_batch, bld_batch = tools.to_cuda(img_batch.permute(1,0,4,2,3)), tools.to_cuda(lbl_batch), tools.to_cuda(bld_batch.permute(1,0,2,3))

        optimizer.zero_grad()
        output, segm1, segm2 =model(img_batch.float())
        output_conf, target_conf = tools.conf_m(output, lbl_batch)
        confusion_matrix.add(output_conf, target_conf)

################Calculate Losses#####################
        sum = output + segm1
        diff = segm2 - segm1
        mychange = bld_batch[-1] - bld_batch[0]
        mychange[mychange==-1]=0

        loss_ch=criterion_ch(output, lbl_batch.long())
        loss_segm1 = criterion_segm(segm1, bld_batch[0].long())
        loss_segm2 = criterion_segm(segm2, bld_batch[-1].long())
        loss_sum = criterion_segm(sum, bld_batch[-1].long())
        loss_diff = criterion_diff(diff, mychange.long())

        loss = 0.8*loss_ch + 0.05*loss_segm1 + 0.05*loss_segm2  + 0.05*loss_sum + 0.05*loss_diff
#######################################

        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        _, preds = output.data.max(1)
        if iter_ % 100 == 0:
         pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
         gt = lbl_batch.data.cpu().numpy()[0]
         print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss_CH: {:.6f}\tAccuracy: {}'.format(
                      epoch, epochs, i, len(mydataset),100.*i/len(mydataset), loss.item(), tools.accuracy(pred, gt)))
        iter_ += 1

    train_acc=(np.trace(confusion_matrix.conf)/float(np.ndarray.sum(confusion_matrix.conf))) *100
    print('TRAIN_LOSS: ', '%.3f' % np.mean(train_losses), 'TRAIN_ACC: ', '%.3f' % train_acc)
    confusion_matrix.reset()

    with torch.no_grad():
        model.eval()
        val_losses = []

        for i, batch, in enumerate(tqdm(mydataset_val)):
            img_batch, lbl_batch, bld_batch = batch
            img_batch, lbl_batch, bld_batch = tools.to_cuda(img_batch.permute(1,0,4,2,3)), tools.to_cuda(lbl_batch), tools.to_cuda(bld_batch.permute(1,0,2,3))

            output, segm1, segm2 =model(img_batch.float())
            output_conf, target_conf = tools.conf_m(output, lbl_batch)
            confusion_matrix.add(output_conf, target_conf)

            sum = output + segm1
            diff = segm2 - segm1
            mychange = bld_batch[-1] - bld_batch[0]
            mychange[mychange==-1]=0

            loss_ch=criterion_ch(output, lbl_batch.long())
            loss_segm1 = criterion_segm(segm1, bld_batch[0].long())
            loss_segm2 = criterion_segm(segm2, bld_batch[-1].long())
            loss_sum = criterion_segm(sum, bld_batch[-1].long())
            loss_diff = criterion_diff(diff, mychange.long())

            loss = 0.8*loss_ch + 0.05*loss_segm1 + 0.05*loss_segm2  + 0.05*loss_sum + 0.05*loss_diff

            val_losses.append(loss.item())

        print(confusion_matrix.conf)
        test_acc=(np.trace(confusion_matrix.conf)/float(np.ndarray.sum(confusion_matrix.conf)))*100
        change_acc=confusion_matrix.conf[1,1]/float(confusion_matrix.conf[1,0]+confusion_matrix.conf[1,1])*100
        non_ch=confusion_matrix.conf[0,0]/float(confusion_matrix.conf[0,0]+confusion_matrix.conf[0,1])*100
        print('VAL_LOSS: ', '%.3f' % np.mean(val_losses), 'VAL_ACC:  ', '%.3f' % test_acc, 'Non_ch_Acc: ', '%.3f' % non_ch, 'Change_Accuracy: ', '%.3f' % change_acc)
        confusion_matrix.reset()


    tools.write_results(ff, save_folder, epoch, train_acc, test_acc, change_acc, non_ch, np.mean(train_losses), np.mean(val_losses))

    #save model in every epoch
    torch.save(model.state_dict(), './' + save_folder + '/model_{}.pt'.format(epoch))
