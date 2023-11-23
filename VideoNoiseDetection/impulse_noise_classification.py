import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_metrics(tp,tn,fp,fn):
    acc = (tp + tn)/(tp+tn+fp+fn)
    recall = tp/(tp+fn)
    prec = tp/(tp+fp)
    f1 = 2*tp/(2*tp+fp+fn)
    return acc, recall, prec, f1

def result(model, test_loader):
    model.eval()
    ssim_without_gt = []
    ssim_wrt_gt = []
    with torch.no_grad():
        psnr_nsy = 0.0
        psnr_de_nsy = 0.0
        ssim_nsy = 0.0
        ssim_de_nsy = 0.0
        
        for X_test, y_test in test_loader:
            X_test = X_test.cuda()
            y_test = y_test.cuda()
            
            y_val = model(X_test)
            # transform to image to compute PSNR
    
            for GT, NSY, PRED in zip(y_test, X_test, y_val):
                gt = np.transpose(GT.cpu().numpy(),(1,2,0))
                nsy = np.transpose(NSY.cpu().numpy(),(1,2,0))
                pred = np.transpose(PRED.cpu().numpy(),(1,2,0))
                
                psnr_nsy += psnr(gt,nsy,data_range=nsy.max() - nsy.min())
                psnr_de_nsy += psnr(gt,pred,data_range=pred.max() - pred.min())
                ssim_nsy += ssim(gt,nsy,data_range=nsy.max() - nsy.min(),channel_axis=-1)
                ssim_de_nsy += ssim(gt,pred,data_range=pred.max() - pred.min(),channel_axis=-1)
                ssim_without_gt.append(ssim(pred,nsy,data_range=pred.max() - pred.min(),channel_axis=-1))
                ssim_wrt_gt.append(ssim(pred,gt,data_range=pred.max() - pred.min(),channel_axis=-1))
        dataset_size = 512
        psnr_nsy = psnr_nsy/dataset_size
        psnr_de_nsy = psnr_de_nsy/dataset_size
        ssim_nsy = ssim_nsy/dataset_size
        ssim_de_nsy = ssim_de_nsy/dataset_size
    
    print('PSNR before denoising :', psnr_nsy)
    print('PSNR after denoising :', psnr_de_nsy)
    print('SSIM before denoising :', ssim_nsy)
    print('SSIM after denoising :', ssim_de_nsy)
    
    ssim_without_gt_np = np.array(ssim_without_gt)
    ssim_wrt_gt_np = np.array(ssim_wrt_gt)
    max_acc = -1
    max_acc_th = -1
    
    for threshold in np.arange(1,0,-0.01):
        tn = len(ssim_without_gt_np[ssim_without_gt_np < threshold])
        tp = len(ssim_wrt_gt_np[ssim_wrt_gt_np > threshold])
        fn = len(ssim_wrt_gt_np[ssim_wrt_gt_np < threshold])
        fp = len(ssim_without_gt_np[ssim_without_gt_np > threshold])
        accuracy = (tp + tn)/(tp+tn+fp+fn)
        if accuracy >= max_acc:
            max_acc = accuracy
            max_acc_th = np.round(threshold,2)
    
    tn = len(ssim_without_gt_np[ssim_without_gt_np < max_acc_th])
    tp = len(ssim_wrt_gt_np[ssim_wrt_gt_np > max_acc_th])
    fn = len(ssim_wrt_gt_np[ssim_wrt_gt_np < max_acc_th])
    fp = len(ssim_without_gt_np[ssim_without_gt_np > max_acc_th])
    accuracy, recall, precision, f1 = calc_metrics(tp,tn,fp,fn)
    print("Threshold: ", np.round(max_acc_th,2))
    print("Accuracy: ",accuracy)
    print("Recall: ",recall)
    print("Precision: ",precision)
    print("F1: ",f1)