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
  
5. Download the provided '/code/' folder and run main.py to begin training. Inside the script you should provide your '/Fsplit/' folder destination as well as the '/xys/' folder destination
```ruby
train_areas = np.load('..../Fsplit/Ftrain.npy').tolist()
val_areas = np.load('..../Fsplit/Fval.npy').tolist()

csv_file_train = '..../xys/myxys_train.csv'
csv_file_val = '..../xys/myxys_val.csv'
```
Also, you should provide the patch size you wish to use when defining the model:
```ruby
model = tools.to_cuda(network.U_Net(4,2,32)) #here 4 is the number of input channels, 2 is the number of output categories (change or no change)
                                             # and 32 is the employed patch size
```                                             
Notice that the patch size should be the same as defined in step 4.
After training, a folder named '/models/' will have been created, where the models from the different epochs will have been saved, as well as a 'progress.txt' file where the accuracies and losses are monitored.

6. Use inf.py to produce the testing predictions. Inside the script you should provide the trained model that you want to use.
```ruby
model.load_state_dict(torch.load('./models/model_9.pt'))
```

