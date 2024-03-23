import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from itertools import chain

class CustomDataset(Dataset):
    def __init__(self, img_dir, annotations, transform=None):
        """
        Naive implementation:
        List all the images in the given directory (trainset/testset)
        
        For all subfolder in root:
            For each image in the subfolder:
                Add image path to the list
        """
        self.img_dir = img_dir
        self.annotations = pd.read_csv(annotations)
        self.transform = transform
        
        self.img_paths = [glob.glob(os.path.join(img_dir + '/' + subfolder, '*.jpg')) for subfolder in os.listdir(img_dir)]
        self.img_paths = list(chain.from_iterable(self.img_paths))
        
        
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        """
        Read the image, extract the pid from the image path
        Locate the corresponding label, gender, lymph_count
        And return all of them
        
        To do: what to do with DOB
        options: year, or the unix age
        """
        image = read_image(self.img_paths[idx])
        PID = self.img_paths[idx].split('/')[-2]
        label = list(self.annotations[self.annotations['ID'] == PID]['LABEL'])[0]
        gender = list(self.annotations[self.annotations['ID'] == PID]['GENDER'])[0]
#         print(type(gender)) -- bug: it should be string, but the dataloader returns a tuple ('M', )
        lymph_count = list(self.annotations[self.annotations['ID'] == PID]['LYMPH_COUNT'])[0]
        
        if self.transform:
            image = self.transform(image)
        
        return image, PID, label, gender, lymph_count