import numpy as np
from skimage import io
from skimage.transform import rotate, resize
import os
import cv2
import pandas as pd
import glob
import random

patch_s = 32 #define the desired patch size
step = 16 #define the step that will be used to extract the patches

Ftrain = np.load('..../Fsplit/Ftrain.npy').tolist()
Fval = np.load('..../Fsplit/Fval.npy').tolist()

def shuffle(vector):
  vector = np.asarray(vector)
  p=np.random.permutation(len(vector))
  vector=vector[p]
  return vector

def sliding_window_train(i_city, labeled_areas, label, window_size, step):
    city=[]
    fpatches_labels=[]

    x=0
    while (x!=label.shape[0]):
     y=0
     while(y!=label.shape[1]):

               if (not y+window_size > label.shape[1]) and (not x+window_size > label.shape[0]):
                line=np.array([x,y, labeled_areas.index(i_city)]) 
                city.append(line)

                new_patch_label = label[x:x + window_size, y:y + window_size]
                ff=np.where(new_patch_label==1)
                #if there are change pixels in the patch, move with a stride equal to step, else 
                #move with a stride equal to window_size(patch_size)
                if ff[0].shape[0]==0:
                       stride=window_size
                else:
                       stride=step

               if y + window_size == label.shape[1]:
                  break

               if y + window_size > label.shape[1]:
                y = label.shape[1] - window_size
               else:
                y = y+stride

     if x + window_size == label.shape[0]:
        break

     if x + window_size > label.shape[0]:
       x = label.shape[0] - window_size
     else:
      x = x+stride

    return np.asarray(city)


if os.path.exists('xys'):
    shutil.rmtree('xys')
os.mkdir('xys')

cities=[]
for i_city in Ftrain:
 print('train ', i_city)
 path = i_city + '/change/change.tif'
 train_gt = io.imread(path)


 xy_city =  sliding_window_train(i_city, Ftrain, train_gt, patch_s, step)
 cities.append(xy_city)

final_cities = np.concatenate(cities, axis=0)
final_cities=shuffle(final_cities)

##save train xys to csv file
df = pd.DataFrame({'X': list(final_cities[:,0]),
                   'Y': list(final_cities[:,1]),
                   'image_ID': list(final_cities[:,2]),
                   })
df.to_csv('./xys/myxys_train.csv', index=False, columns=["X", "Y", "image_ID"])


cities=[]
for i_city in Fval:
 print('val ', i_city)
 path = i_city + '/change/change.tif'
 val_gt = io.imread(path)

 xy_city =  sliding_window_train(i_city, Fval, val_gt, patch_s, patch_s)
 cities.append(xy_city)

final_cities = np.concatenate(cities, axis=0)
final_cities=shuffle(final_cities)

##save val xys to csv file
df = pd.DataFrame({'X': list(final_cities[:,0]),
                   'Y': list(final_cities[:,1]),
                   'image_ID': list(final_cities[:,2]),
                   })
df.to_csv('./xys/myxys_val.csv', index=False, columns=["X", "Y", "image_ID"])
