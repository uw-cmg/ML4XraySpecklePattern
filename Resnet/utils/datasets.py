# import the necessary packages
import scipy.io as sio
import numpy as np
import glob
import os

def load_data_std(inputPath, nb_classes, nb_img_per_class,img_rows, img_cols):
    X_raw = []
    Y_raw = []
    for i in range(nb_classes):
        for j in range(nb_img_per_class):
            img_rgb = np.zeros((img_rows, img_cols,3))
            mat_filename = os.path.sep.join([inputPath, "SpeckleBeads_" + str(i+2),"Beads_"+ str(i+2)+"."+str(j)+".mat"])
            mat_Acontents = sio.loadmat(mat_filename)
            img = mat_Acontents['pattern_data']
            assert img.shape == (img_rows, img_cols) # (513 x 513)
            # calculate global mean and standard deviation
            mean, std = img.mean(), img.std()
            # global standardization of pixels
            img = (img - mean) / std
            # duplicate to RGB like 3 channels
            img_rgb[:,:,0] = img
            img_rgb[:,:,1] = img
            img_rgb[:,:,2] = img
            X_raw.append(img_rgb)
            Y_raw.append(i)
    return X_raw, Y_raw

def load_data_norm(inputPath, nb_classes, nb_img_per_class,img_rows, img_cols):
    X_raw = []
    Y_raw = []
    for i in range(nb_classes):
        for j in range(nb_img_per_class):
            img_rgb = np.zeros((img_rows, img_cols,3))
            mat_filename = os.path.sep.join([inputPath, "SpeckleBeads_" + str(i+2),"Beads_"+ str(i+2)+"."+str(j)+".mat"])
            mat_Acontents = sio.loadmat(mat_filename)
            img = mat_Acontents['pattern_data']
            assert img.shape == (img_rows, img_cols) # (513 x 513)
            img_min, img_max = img.min(), img.max()
            img = (img - img_min) / (img_max - img_min)
            img_rgb[:,:,0] = img
            img_rgb[:,:,1] = img
            img_rgb[:,:,2] = img
            X_raw.append(img)
            Y_raw.append(i)
    return X_raw, Y_raw

def load_data_norm_std(inputPath, nb_classes, nb_img_per_class,img_rows, img_cols):
    X_raw = []
    Y_raw = []
    for i in range(nb_classes):
        for j in range(nb_img_per_class):
            img_rgb = np.zeros((img_rows, img_cols,3))
            mat_filename = os.path.sep.join([inputPath, "SpeckleBeads_" + str(i+2),"Beads_"+ str(i+2)+"."+str(j)+".mat"])
            mat_Acontents = sio.loadmat(mat_filename)
            img = mat_Acontents['pattern_data']
            assert img.shape == (img_rows, img_cols) # (513 x 513)
            # normalization
            img_min, img_max = img.min(), img.max()
            img = (img - img_min) / (img_max - img_min)
            # Standardization
            # calculate global mean and standard deviation
            mean, std = img.mean(), img.std()
            img = (img - mean) / std
            img_rgb[:,:,0] = img
            img_rgb[:,:,1] = img
            img_rgb[:,:,2] = img
            X_raw.append(img)
            Y_raw.append(i)
    return X_raw, Y_raw
######################################
# Code below is for MD speckle pattern
######################################
def calculate_angular_distribution(speckle_pattern : list) -> list:
    # get properties of speckle image
    lenX = speckle_pattern[0].shape[0]
    lenY = speckle_pattern[0].shape[1]
    center_X = lenY // 2
    center_Y = lenY // 2
    # output angular distribution
    speckle_accumlation = []
    # Loop throught the accumlation list for each cell using the center to calculate the accumlation
    for img in speckle_pattern:
        img_accumlation = list()
        for i in range(int(np.sqrt((0 - center_X)**2 + (0 - center_Y)**2)) + 1):
            img_accumlation.append( [] )
        for i in range(lenX):
            for j in range(lenY):
                ind = int(np.sqrt((i - center_X)**2 + (j - center_Y)**2))
                assert ( 0 <= ind <= (int(np.sqrt((0 - center_X)**2 + (0 - center_Y)**2)) + 1) )
                img_accumlation[ind].append(img[i,j,0])
        img_angular = list()
        for acc in img_accumlation:
            img_angular.append(np.mean(acc))
        speckle_accumlation.append(img_angular)
    return speckle_accumlation

def radAvgToSpeckleImage(radAvgList):
    speckle_patterns = []
    center_X = 50
    center_Y = 50
    # Loop through all frames
    for frame in radAvgList:
        # Create image of speckle pattern
        img = np.zeros((100,100,1))
        # Loop trough all image cells
        for i in range(100):
            for j in range(100):
                ind = int(np.sqrt((i - center_X)**2 + (j - center_Y)**2))
                img[i,j,0] = frame[ind]
				#img[i,j,1] = frame[ind]
				#img[i,j,2] = frame[ind]
        speckle_patterns.append(img)
    return speckle_patterns

def radAvgToSpeckleImage3D(radAvgList):
    speckle_patterns = []
    center_X = 50
    center_Y = 50
    # Loop through all frames
    for frame in radAvgList:
        # Create image of speckle pattern
        img = np.zeros((100,100,3))
        # Loop trough all image cells
        for i in range(100):
            for j in range(100):
                ind = int(np.sqrt((i - center_X)**2 + (j - center_Y)**2))
                img[i,j,0] = frame[ind]
                img[i,j,1] = frame[ind]
                img[i,j,2] = frame[ind]
        speckle_patterns.append(img)
    return speckle_patterns
