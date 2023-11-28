import os
import pandas as pd
import cv2
from patchify import patchify
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

def patches(img,patch_size):
    patches = patchify(img, (patch_size, patch_size, 3), step=patch_size)
    return patches

def get_dataloaders(root='./VideoNoiseDetection/datasets/SIDD_Small_sRGB_Only/Data/'):
    dir = os.listdir(root[:-1])
    folders=[]
    for folder in dir:
      folders.append(folder)
    
    GT = []
    Noisy = []
    for folder in folders:
        files = os.listdir(root+folder)
        for img in files:
            if img[0]=='G':
                GT.append(root+folder+'/'+img)
            else:
                Noisy.append(root+folder+'/'+img)
    
    df = pd.DataFrame()
    df['Ground Truth Images'] = GT
    df['Noisy Images'] = Noisy
    size=[]
    for i in range(len(df)):
        img_gt = cv2.imread(df['Ground Truth Images'].iloc[i])
        size.append(img_gt.shape)
    df['image size'] = size
    df['image size'] = df['image size'].astype(str)
    
    X = df['Noisy Images']
    y = df['Ground Truth Images']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Creating patches for X_train and y_train
    X_train_patches = []
    y_train_patches = []
    for i in range(len(X_train)):
        path = X_train.iloc[i]
        img_nsy = cv2.imread(path)
        img_nsy = cv2.cvtColor(img_nsy, cv2.COLOR_BGR2RGB)
        img_nsy = cv2.resize(img_nsy,(1024,1024))
        patches_nsy = patches(img_nsy,256)
        
        path = y_train.iloc[i]
        img_gt = cv2.imread(path)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_gt = cv2.resize(img_gt,(1024,1024))
        patches_gt = patches(img_gt,256)
        
        rows = patches_nsy.shape[0]
        cols = patches_nsy.shape[1]
        for j in range(rows):
            for k in range(cols):
                X_train_patches.append(patches_nsy[j][k][0])
                y_train_patches.append(patches_gt[j][k][0])
                
        if (i + 1) % 10 == 0:
            print(f"X_train: {i+1} images processed")
      
    X_train_patches = np.array(X_train_patches)
    y_train_patches = np.array(y_train_patches)
    
    # Creating patches for X_test and y_test
    X_test_patches = []
    y_test_patches = []
    for i in range(len(X_test)):
        path = X_test.iloc[i]
        img_nsy = cv2.imread(path)
        img_nsy = cv2.cvtColor(img_nsy, cv2.COLOR_BGR2RGB)
        img_nsy = cv2.resize(img_nsy,(1024,1024))
        patches_nsy = patches(img_nsy,256)
        
        path = y_test.iloc[i]
        img_gt = cv2.imread(path)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_gt = cv2.resize(img_gt,(1024,1024))
        patches_gt = patches(img_gt,256)
        
        rows = patches_nsy.shape[0]
        cols = patches_nsy.shape[1]
        for j in range(rows):
            for k in range(cols):
                X_test_patches.append(patches_nsy[j][k][0])
                y_test_patches.append(patches_gt[j][k][0])

        if (i + 1) % 10 == 0:
            print(f"X_test: {i+1} images processed")
    
    X_test_patches = np.array(X_test_patches)
    y_test_patches = np.array(y_test_patches)
    
    X_train_patches = np.transpose(X_train_patches, (0,3,1,2))
    y_train_patches = np.transpose(y_train_patches, (0,3,1,2))
    X_test_patches = np.transpose(X_test_patches, (0,3,1,2))
    y_test_patches = np.transpose(y_test_patches, (0,3,1,2))
    
    X_train_patches = X_train_patches.astype("float32") / 255.0
    y_train_patches = y_train_patches.astype("float32") / 255.0
    X_test_patches = X_test_patches.astype("float32") / 255.0
    y_test_patches = y_test_patches.astype("float32") / 255.0
    
    train_patches = TensorDataset(torch.Tensor(X_train_patches),torch.Tensor(y_train_patches))
    test_patches = TensorDataset(torch.Tensor(X_test_patches),torch.Tensor(y_test_patches))
    batch_size=32
    train_loader = DataLoader(train_patches, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_patches, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader