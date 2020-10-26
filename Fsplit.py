import numpy as np
import os
import glob
import random
import shutil

FOLDER = glob.glob('..../train/*_13*') #give your '/train/' folder destination
random.shuffle(FOLDER)
Ftrain = FOLDER[0:40]
Fval = FOLDER[40:50]
Ftest = FOLDER[50:60]


if os.path.exists('Fsplit'):
    shutil.rmtree('Fsplit')
os.mkdir('Fsplit')
np.save('./Fsplit/Ftrain.npy', Ftrain)
np.save('./Fsplit/Fval.npy', Fval)
np.save('./Fsplit/Ftest.npy', Ftest)


print(len(Ftrain), 'folders for training')
print(len(Fval), 'folders for validation')
print(len(Ftest), 'folders for testing')
