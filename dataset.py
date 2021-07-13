from PIL import Image, ImageOps
import os
import numpy as np
from torch.utils.data import Dataset

class GlassesDataset(Dataset):
    def __init__(self, root_dir_glass, root_dir_noglass, transform=None):
        super().__init__()
        self.root_dir_glasses = root_dir_glass
        self.root_dir_noglasses = root_dir_noglass
        self.list_glasses = os.listdir(self.root_dir_glasses)
        self.list_noglasses = os.listdir(self.noroot_dir_glasses)
        self.transform = transform
        self.len_glasses = len(self.list_glasses)
        self.len_noglasses = len(self.list_noglasses)
        self.dataset_length = max(self.len_glasses, self.len_noglasses)
        
    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, index):
        glasses_file = self.list_glasses[index % self.len_glasses]
        no_glasses_file = self.list_noglasses[index % self.len_noglasses]
        glasses_path = os.path.join(self.root_dir_glasses, glasses_file)
        no_glasses_path = os.path.join(self.root_dir_noglasses, no_glasses_file)
        img_glasses = np.array(ImageOps.grayscale(Image.open(glasses_path)))
        img_noglasses = np.array(ImageOps.grayscale(Image.open(no_glasses_path)))
        
        if self.transform:
            augmentation = self.transform(image=img_glasses, image0=img_noglasses)
            img_glasses = augmentation["image"]
            img_noglasses = augmentation["image0"]
            
        return img_glasses, img_noglasses
