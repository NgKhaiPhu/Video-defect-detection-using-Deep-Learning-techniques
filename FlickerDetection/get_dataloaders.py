import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import torch
import os

import pytorchvideo
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RemoveKey,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    Resize,
)

def get_data(path):
    video_list = []
    
    for root, dirs, files in os.walk(path):
        for name in files:
            video_list.append(os.path.join(root, name))

    return video_list

class CustomDataset(Dataset):
    def __init__(self, dataset):
            super().__init__()
            self.dataset = dataset
            self.targets = torch.FloatTensor([int(data.split('.')[-2][-1]) for data in self.dataset])
        
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            resize_to = (224, 224)
            num_frames = 16
        
            self.transform =  ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        Lambda(lambda x: x/255.0),
                        Normalize(mean, std),
                        Resize(resize_to,antialias=True)
                    ]
                ),
            )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        video_path = self.dataset[idx]
        label = self.targets[idx]

        video = EncodedVideo.from_path(video_path, decode_audio=False)
        video_data = video.get_clip(start_sec=0,end_sec=0.99)
        video_data = self.transform(video_data)
        inputs = video_data["video"]
    
        return inputs, label

def get_dataloaders(root='./FlickerDetection/datasets/16frames/data'):
    lst = get_data(root)
    df = pd.DataFrame(lst,columns=['path'])
    index = np.random.choice(range(5717), 5717-603, replace=False)
    for i in range(len(df)):
        df.loc[i,'label'] = str(df.loc[i,'path'].split('.')[-2][-1])
    df2 = df[df['label'] == '0'].reset_index(drop = True)
    for i in index:
        df2.drop(index=i,axis=0,inplace=True)
    df2 = pd.concat([df2, df[df['label'] == '1']])
    df2.drop(columns='label',axis=1,inplace=True)
    lst = list(df2['path'])
    
    dataset = CustomDataset(dataset=lst)
    train_size = int(0.8 * len(dataset))
    val_size = int((len(dataset) - train_size) / 2)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    batch_size = 16
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=12)
    val_loader = DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False,num_workers=12)
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=12)

    return train_loader, val_loader, test_loader