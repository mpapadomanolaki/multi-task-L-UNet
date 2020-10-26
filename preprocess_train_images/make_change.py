import os
from skimage import io
import numpy as np
import cv2
import glob
import shutil

L = glob.glob('*_13*')

cnt = 0

for c in range (0, len(L)):
    l = L[c]
    print(c ,'/' ,len(L), ' foders processed')
    if os.path.exists('./{}/change'.format(l)):
        shutil.rmtree('./{}/change'.format(l))
    os.mkdir('./{}/change'.format(l))


    b1 = io.imread('./{}/buildings/buildings1.tif'.format(l))
    b2 = io.imread('./{}/buildings/buildings2.tif'.format(l))


    b2[np.where(b1==1)] = 0
    cv2.imwrite('./' + l + '/change/change.tif', b2)

    cnt = cnt + 1
