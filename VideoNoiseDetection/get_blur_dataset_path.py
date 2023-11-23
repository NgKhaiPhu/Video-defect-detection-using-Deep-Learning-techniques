import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd

def get_index(path):
    return int(path.split('/')[-1].split('_')[0])\

def get_blur_dataset_path(df, dataset_path=None):
    temp_list = []
    if dataset_path == None:
        print("Must include folder!")
        return
    
    for root, dirs, files in os.walk(dataset_path):
        if root != dataset_path:
            current_folder = []
            for file in files:
                current_folder.append(os.path.join(root,file))
            current_folder.sort()
            temp_list.append(current_folder)

    if len(temp_list) != 3:
        print("Error reading folder !!!\n1) Ensure you have installed blur_dataset to the correct path. \n2) dataset_path needs to be relative to main.py.")
        return
        
    for path in temp_list[0]:
        df.loc[get_index(path),'defocus_blur'] = path
    for path in temp_list[1]:
        df.loc[get_index(path),'sharp'] = path
    for path in temp_list[2]:
        df.loc[get_index(path),'motion_blur'] = path

    return df