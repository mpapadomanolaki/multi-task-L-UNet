import numpy as np
import os
import glob
import random


FOLDER = glob.glob('../train/*_13*')
random.shuffle(FOLDER)
Ftrain = FOLDER[0:40]
Fval = FOLDER[40:50]
Ftest = FOLDER[50:60]
np.save('./Fsplit/Ftrain.npy', Ftrain)
np.save('./Fsplit/Fval.npy', Fval)
np.save('./Fsplit/Ftest.npy', Ftest)
