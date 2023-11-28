import numpy as np
from glitch_this import ImageGlitcher
import os
import PIL
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms

def get_images(path):
    images_list = []
    
    for root, dirs, files in os.walk(path):
        for name in files:
            images_list.append(os.path.join(root, name))

    return images_list

class CustomDataset(Dataset):
    def __init__(self, root, target, transform=None):
        self.root = root
        self.images = get_images(root)
        self.targets = [int(target) for i in range(len(self.images))]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image_name = self.images[index]  
        image = PIL.Image.open(image_name).convert('RGB')
        label = torch.tensor(self.targets[index],dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return (image, label)

def pixelate(img):
    imgSmall = img.resize((128,128), resample=PIL.Image.Resampling.BILINEAR)
    return imgSmall.resize(img.size, PIL.Image.Resampling.NEAREST)

def shuffle_pixel(region):
    temp = np.reshape(region, (-1, region.shape[2]))
    np.random.shuffle(temp)
    region = np.reshape(temp, region.shape)
    return region

def distort(img):
    img = np.array(img)
    for i in range(img.shape[0]//14):
        for j in range(img.shape[1]//14):
            if np.random.randint(0,5) == 0:
                img[14*i:14*i+14,14*j:14*j+14,:] = shuffle_pixel(img[14*i:14*i+14,14*j:14*j+14,:])

    return PIL.Image.fromarray(np.uint8(img)).convert('RGB')

def glitch(img):
    glitcher = ImageGlitcher()
    glitch_amount = np.random.randint(2,9)
    if np.random.randint(0,1) == 0:
        return glitcher.glitch_image(img, glitch_amount, color_offset=False)
    else:
        return glitcher.glitch_image(img, glitch_amount, color_offset=True)
    
# Main augmentation function
def noise_lambda(img):
    if np.random.randint(0,2) == 0:
        return pixelate(distort(img))
    else:
        return glitch(img)

def crop_black_border(image):
    y_nonzero, _, _ = np.nonzero(np.array(image) > 10)
    return image.crop((0, np.min(y_nonzero), image.size[0], np.max(y_nonzero)))

# UCF101 dataset
def get_dataloaders(root='./VideoNoiseDetection/datasets/505_video_frames/'):
    batch_size=32
    normal_transform = transforms.Compose([
            transforms.Lambda(crop_black_border),
            transforms.Resize((224,224)),           
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    glitch_transform = transforms.Compose([ 
            transforms.Lambda(crop_black_border),
            transforms.Resize((224,224)),  
            transforms.Lambda(noise_lambda),  
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    
    original_dataset = CustomDataset(root=root,target=0,transform=normal_transform)
    glitch_dataset = CustomDataset(root=root,target=1,transform=glitch_transform)
    dataset = ConcatDataset([original_dataset, glitch_dataset])

    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader