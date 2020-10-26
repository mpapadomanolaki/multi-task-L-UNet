# multi-task-L-UNet

This repository includes the code for the following manuscript:

'A Deep Multi-Task Learning Framework Coupling Semantic Segmentation and Fully Convolutional LSTM Networks for Urban Change Detection'

The proposed method has been implemented on the training images of SpaceNet7 dataset.

## Steps

1. Download the training images of SpaceNet7 dataset as described in: https://github.com/CosmiQ/CosmiQ_SN7_Baseline
   Rename the downloaded folder '/SN7_buildings_train/' to '/train/'. This folder contains all 60 folders of SpaceNet7 training images.
   
2. Preprocess data
   - Inside the /train/ folder put the the scripts included in the provided '/preprocess_train_images/' folder (binary_mask.py, make_buildings.py, make_change.py)
   - Run make_buildings.py to create building binary masks for the first and the last date of every folder
   - Run make_change.py to create the change binary mask for every folder 

3. Run Fsplit.py to split the training folders to training, validation and testing parts. Inside this script you should provide your destination path of the '/train/' folder: 
```ruby
    FOLDER = glob.glob('..../train/*_13*')
```
The folder identity paths will be saved as numpy arrays in a folder named '/Fsplit/'.
   
4. Run create_xys.py to create csv files with the xy coordinates that will be used to extract the patches during training and validation. Inside the script you should set the patch size that you want, as well as the step that wil be used to extract the patches along the x and y dimensions. Provide also your '/Fsplit/' folder destination:
```ruby
    patch_s = 32 #define the desired patch size
    step = 16 #define the step that will be used to extract the patches

    Ftrain = np.load('..../Fsplit/Ftrain.npy').tolist()
    Fval = np.load('..../Fsplit/Fval.npy').tolist()  
```
   The csv files will be saved in a folder named '/xys/'. 
  
5. Download the provided '/code/' folder and run main.py to begin training. Inside the script you should provide your '/Fsplit/' folder destination path as well as the '/xys/' folder destination path:
```ruby
    train_areas = np.load('..../Fsplit/Ftrain.npy').tolist()
    val_areas = np.load('..../Fsplit/Fval.npy').tolist()

    csv_file_train = '..../xys/myxys_train.csv'
    csv_file_val = '..../xys/myxys_val.csv'
```
Also, you should provide the patch size you want to use and the number of available dates:
```ruby
    patch_size=32
    nb_dates=19
```
Notice that the patch size should be the same as defined in step 4. Here the number of dates is equal to 19, because in the available SpaceNet7 training images, 19 dates is the minimum number of dates provided for a folder. Hence, we want to provide the same number of dates for each of the folders.
                                          
After training, a folder named '/models/' will have been created, where the models from the different epochs will have been saved, as well as a 'progress.txt' file where the accuracies and losses are monitored.

6. Use inf.py to produce the testing predictions. Inside the script you should provide the trained model that you want to use:
```ruby
    model.load_state_dict(torch.load('./models/model_.....pt'))
```
As well as your '/Fsplit/' folder destination path:
```ruby
    FOLDER = np.load('..../Fsplit/Ftest.npy').tolist()
```
The final predictions as well as the probability maps will be saved in a folder named '/PREDICTIONS/'.

Notice: Here the experiments are performed using 10 dates. Specifically, in the custom.py script, there is the following part:
```ruby
    for nd in range(0, nb_dates-1, 2): #    
        im = io.imread(sort_tifs[nd])
        img.append(im)
    img.append( io.imread(sort_tifs[-1]) )
    self.all_imgs.append(np.asarray(img))
```
If you notice the for loop, you will observe that we iterate through the 19 dates with a step of 2. After the for loop, we take also the last available date for each folder. In other words, we utilize the first and the last date, as well as 8 intermediate dates. For using a different number of dates, you can change the step in the for loop.

If you find this work useful, please consider citing:


