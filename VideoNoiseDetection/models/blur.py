import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from skimage.measure import blur_effect
from skimage.feature import canny
from skimage.filters import sobel

import sys
sys.path.append('../datasets/blur')
from blur_dataset import get_blur_dataset

def fix_image_size(image,expected_pixels=2e6):
    ratio = np.sqrt(expected_pixels/(image.shape[0]*image.shape[1]))
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)

def estimate_blur(image):
    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = np.var(blur_map)
    return score

df = pd.DataFrame()
df = get_blur_dataset(df)
blur_df = pd.DataFrame()
for col in df.columns:
    for index, img_path in enumerate(df[col]):
        img = cv2.imread(img_path,0)
        img = fix_image_size(img)
        
        blur_df.loc[index, col] = estimate_blur(img)

max_acc = -1
max_acc_th = -1

def calc_metrics(tp,tn,fp,fn):
    acc = (tp + tn)/(tp+tn+fp+fn)
    recall = tp/(tp+fn)
    prec = tp/(tp+fp)
    f1 = 2*tp/(2*tp+fp+fn)
    return acc, recall, prec, f1

for threshold in np.arange(0,1000,1):
    tn = len(blur_df[blur_df['defocus_blur'] < threshold])
    tp = len(blur_df[blur_df['sharp'] > threshold])
    fn = len(blur_df[blur_df['sharp'] < threshold])
    fp = len(blur_df[blur_df['defocus_blur'] > threshold])
    accuracy = (tp + tn)/(tp+tn+fp+fn)
    if accuracy >= max_acc:
        max_acc = accuracy
        max_acc_th = np.round(threshold,2)

tn = len(blur_df[blur_df['defocus_blur'] < max_acc_th])
tp = len(blur_df[blur_df['sharp'] > max_acc_th])
fn = len(blur_df[blur_df['sharp'] < max_acc_th])
fp = len(blur_df[blur_df['defocus_blur'] > max_acc_th])
accuracy, recall, precision, f1 = calc_metrics(tp,tn,fp,fn)
print("Threshold: ",np.round(max_acc_th,2))
print("Accuracy: ",accuracy)
print("Recall: ",recall)
print("Precision: ",precision)
print("F1: ",f1)