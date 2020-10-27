# multi-task-L-UNet

This repository includes the code for the following manuscript:

'A Deep Multi-Task Learning Framework Coupling Semantic Segmentation and Fully Convolutional LSTM Networks for Urban Change Detection'

The proposed method has been implemented on the training images of SpaceNet7 dataset.

Apart from the proposed method, you can also use the 'preprocess' folder to create binary masks for the SpaceNet7 dataset.

## Steps

1. Download the training images of SpaceNet7 dataset as described in: https://github.com/CosmiQ/CosmiQ_SN7_Baseline
   This folder contains all 60 folders of SpaceNet7 training images.
   
2. Preprocess data
   - Use the scripts in the 'preprocess_train_images' folder to preprocess the SpaceNet7 training images.
   - Run make_buildings.py to create building binary masks for the first and the last date of the available SpaceNet7 areas. This script will create a 'buildings'
     subfolder for every area where the binary masks will be saved.
   - In the same way, run make_change.py to create the change binary mask for every folder. 

3. Run Fsplit.py to split the training folders to training, validation and testing parts. The folder identity paths will be saved as numpy arrays in a folder named
   '/Fsplit/'.
   
4. Run create_xys.py to create csv files with the xy coordinates that will be used to extract the patches during training and validation. Inside the script you
   should set the patch size that you want, as well as the step that wil be used to extract the patches along the x and y dimensions. Provide also your '/Fsplit/'   
   folder destination from step 3:
   The csv files will be saved in a folder named '/xys/'. 
  
5. Download the provided '/code/' folder and run main.py to begin training. Inside the script you should provide your '/Fsplit/' folder destination path as well as
   the '/xys/' folder destination path:   
   Also, you should provide the patch size you want to use and the number of available dates:

   Notice that the patch size should be the same as defined in step 4. Here the number of dates is equal to 19, because in the available SpaceNet7 training images,  
   the available dates in each folder range from 19 to 24. Since we want to provide the same number of dates for each of the folders, we utilize 19 dates from each
   folder.
                                          
    After training, a folder named '/models/' will have been created, where the models from the different epochs will have been saved, as well as a 'progress.txt' 
    file where the accuracies and losses are monitored.

6. Use inf.py to produce the testing predictions. Inside the script you should provide the trained model that you want to use. The final predictions as well as the probability maps will be saved in a folder named '/PREDICTIONS/'.

_Notice_: Here the experiments are performed using 10 dates. Specifically, in the custom.py script, there is the following part:
```ruby
    for nd in range(0, nb_dates-1, 2):    
        im = io.imread(sort_tifs[nd])
        img.append(im)
    img.append( io.imread(sort_tifs[-1]) )
    self.all_imgs.append(np.asarray(img))
```
If you notice the 'for' loop, you will observe that we iterate through the 19 dates with a step of 2. After the 'for' loop, we take also the last available date for each folder. In other words, we utilize the first and the last date, as well as 8 intermediate dates. For using a different number of dates, you can change the step in the 'for' loop.

If you find this work useful, please consider citing:

M. Papadomanolaki, M. Vakalopoulou, K. Karantzalos, 'A Deep Multi-Task Learning Framework Coupling Semantic Segmentation and Fully Convolutional LSTM Networks for Urban Change Detection', *IEEE Transactions on Geoscience and Remote Sensing*
