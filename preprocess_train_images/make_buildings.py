import os
from skimage import io
import numpy as np
import cv2
import glob
import shutil
import argparse
from binary_mask import generate_mask


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--train_images_folder', type=str, default='./',
                    help='downloaded folder of SpaceNet7 training images')

args = parser.parse_args()

L = glob.glob(args.train_images_folder + '*_13*')
cnt = 0

for c in range (0, len(L)):
    l = L[c]
    print(c ,'/' ,len(L), ' foders processed')
    if os.path.exists('./{}/buildings'.format(l)):
        shutil.rmtree('./{}/buildings'.format(l))
    os.mkdir('./{}/buildings'.format(l))
    jsons = glob.glob('./' + l + '/labels_match/*.geojson*')

    #sort geojson files in chronological order
    years = []
    for j in range(len(jsons)):
      ff = jsons[j].find('monthly_')
      years.append(jsons[j][ff+8:ff+12] + jsons[j][ff+13:ff+15])
    ind = np.argsort(years)
    f_jsons = [jsons[i] for i in ind]

    #Use only first and last date to create the buildings binary mask
    J_builds = [f_jsons[0], f_jsons[-1]]
    if l == args.train_images_folder + 'L15-1716E-1211N_6864_3345_13':
        J_builds = [f_jsons[0], f_jsons[-2]] # for this folder the geojson for the last date does not contain any polygons so we will take
                                             # the second last date

    for j in J_builds:
         name0 = j.find('match/') + 6
         name = j[name0:]
         im_path = './' + l + '/images/' + name[:-18] + '.tif'
         mask = generate_mask(im_path, j)
         cv2.imwrite('./' + l + '/buildings/buildings' +str(J_builds.index(j)+1)+ '.tif', mask)

    cnt = cnt + 1
