import glob
import cv2
from skimage import io
import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import network
import tools
import shutil

def sliding_window(IMAGE, patch_size, step):
    prediction = np.zeros((IMAGE.shape[3], IMAGE.shape[4], 2))
    count_image = np.zeros((IMAGE.shape[3], IMAGE.shape[4]))
    x=0
    while (x!=IMAGE.shape[3]):
     y=0
     while(y!=IMAGE.shape[4]):

               if (not y+patch_size > IMAGE.shape[4]) and (not x+patch_size > IMAGE.shape[3]):
                patch = IMAGE[:, :, :, x:x + patch_size, y:y + patch_size]/255.0
                patch = (torch.from_numpy(patch).float()).cuda(2)
                output, segm1, segm2 = (model(patch))
                output = F.log_softmax(output)
                output = output.cpu().data.numpy().squeeze()
                output = np.transpose(output, (1,2,0))
                for i in range(0, patch_size):
                    for j in range(0, patch_size):
                        prediction[x+i, y+j] += (output[i,j,:])
                        count_image[x+i, y+j] +=1

                stride=step

               if y + patch_size == IMAGE.shape[4]:
                  break

               if y + patch_size > IMAGE.shape[4]:
                y = IMAGE.shape[4] - patch_size
               else:
                y = y+stride

     if x + patch_size == IMAGE.shape[3]:
        break

     if x + patch_size > IMAGE.shape[3]:
       x = IMAGE.shape[3] - patch_size
     else:
      x = x+stride

    final_pred = np.zeros((IMAGE.shape[3], IMAGE.shape[4]))

    for i in range(0, final_pred.shape[0]):
        for j in range(0, final_pred.shape[1]):
            final_pred[i,j] = np.argmax(prediction[i,j]/float(count_image[i,j]))

    for i in range(0, prediction.shape[0]):
        for j in range(0, prediction.shape[1]):
            prediction[i,j] = (prediction[i,j]/float(count_image[i,j]))

    return final_pred, prediction

########

BATCH_SIZE=1
patch_size = 256
step = 128
model = tools.to_cuda(networkL.U_Net(4,2,256))
model.load_state_dict(torch.load('./models/model_9.pt'))
model=model.cuda(2)
model.eval()

FOLDER = np.load('/home/mariapap/DATA/SPACENET7/EXPS/Fsplit/Ftest.npy').tolist()
nb_dates = 19

save_folder = 'PREDICTIONS' #where to save the testing predictions
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.mkdir(save_folder)


cnt = 1
for c in range(0, len(FOLDER)):

    fold = FOLDER[c]

    all_tifs = glob.glob(fold + '/images/*.tif*')
    years = []
    for j in range(len(all_tifs)):
        ff = all_tifs[j].find('monthly_')
        years.append(all_tifs[j][ff+8:ff+12] + all_tifs[j][ff+13:ff+15])
    ind = np.argsort(years)
    sort_tifs = [all_tifs[i] for i in ind]

    img = []
    for nd in range(0, nb_dates-1, 2):
        im = io.imread(sort_tifs[nd])
        img.append(im)
    img.append( io.imread(sort_tifs[-1]) )

    img = np.asarray(img) #(19,1024,1024,4)
    img = np.transpose(img, (0,3,1,2))
    imgs = np.expand_dims(img, 1)
    pred, prob = sliding_window(imgs, patch_size, step)
    prob = np.transpose(prob, (2,0,1))
    io.imsave('./' +save_folder+ '/mL_PRED_' + fold[36:] + '.tif', pred)
    io.imsave('./' +save_folder+ '/mL_PROB_' + fold[36:] + '.tif', prob)
    
    print(cnt, '/', len(FOLDER), 'predictions saved')
    cnt = cnt + 1

