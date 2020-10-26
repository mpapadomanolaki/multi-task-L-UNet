# multi-task-L-UNet

This repository includes the code for the following manuscript:

'A Deep Multi-Task Learning Framework Coupling Semantic Segmentation and Fully Convolutional LSTM Networks for Urban Change Detection'

The proposed method has been implemented on the training images of SpaceNet7 dataset.

## Steps


1. Download the training images of SpaceNet7 dataset as described in: https://github.com/CosmiQ/CosmiQ_SN7_Baseline
   Rename the downloaded folder '/SN7_buildings_train/' to '/train/'. This folder contains all 60 folders of SpaceNet7 training images.
   
2. Preprocess data
   - Inside the /train/ folder put the the scripts included in the provided 'preprocess_train_images' folder (binary_mask.py, make_buildings.py, make_change.py)
   - Run make_buildings.py to create building binary masks for the first and the last date of every folder
   - Run make_change.py to create the change binary mask for every folder 

3. Run Fsplit.py to split the training folders to training, validation and testing parts. The folder identity paths will be saved as numpy arrays in a folder named '/Fsplit/'.
   
4. Run create_xys.py to create csv files with the xy coordinates that will be used to extract the patches during training and validation. The csv files will be saved in a folder named '/xys/'.   
  
5. Download in a folder all scripts of the provided '/code/' folder and run main.py to begin training.

6. Use inf.py to produce the testing predictions.
