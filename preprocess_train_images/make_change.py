import os
from skimage import io
import numpy as np
import cv2
import glob
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_images_folder', type=str, default='./',
                    help='downloaded folder of SpaceNet7 training images')

args = parser.parse_args()

L = glob.glob(args.train_images_folder + '*_13*')
cnt = 0

for c in range (0, len(L)):
    l = L[c]
    print(c ,'/' ,len(L), ' foders processed')
    if os.path.exists(args.train_images_folder + '{}/change'.format(l)):
        shutil.rmtree(args.train_images_folder + '{}/change'.format(l))
    os.mkdir(args.train_images_folder + '{}/change'.format(l))


    b1 = io.imread(args.train_images_folder + '{}/buildings/buildings1.tif'.format(l))
    b2 = io.imread(args.train_images_folder + '{}/buildings/buildings2.tif'.format(l))


    b2[np.where(b1==1)] = 0
    cv2.imwrite(args.train_images_folder + l + '/change/change.tif', b2)

    cnt = cnt + 1
