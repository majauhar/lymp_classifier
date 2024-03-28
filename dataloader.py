import glob
from itertools import chain
import numpy as np
import os
from os import path
import pandas as pd
from pathlib import Path 
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose
from typing import List

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

class BagDataset(Dataset):
    """This dataset returns the images and clinical data for each patient."""

    def __init__(self, img_dir: str, data_df: pd.DataFrame, transform: Compose = None) -> None:
        """
        Args:
            img_dir: Path to the images directory.
            data_df: List of subjects used.
            transform: Optional, transformations applied to the tensor.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.data_df = data_df

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> "dict[List[Tensor], int, str, float, float, float]":
        """
        Args:
            idx: The index of the subject whose data is loaded.
        Returns:
            sample: corresponding data described by the following keys:
                images: List with image tensors.
                label: The diagnosis code.
                id: ID of the participant.
                gender: 0.0 for male, '1' for female.
                age: Age value.
                lymph_count: Lymph count value. 
        """

        label = self.data_df.loc[idx, 'LABEL']
        age = self.data_df.loc[idx, 'AGE']
        gender = self.data_df.loc[idx, 'GENDER']
        lymph_count = self.data_df.loc[idx, 'LYMPH_COUNT']

        id = self.data_df.loc[idx, 'ID']
        folder_name = path.join(self.img_dir, id)
        images = []
        for filename in Path(folder_name).glob('*'):
          image = read_image(str(filename))

          if self.transform:
              image = self.transform(image)
          images.append(image)

        sample = {'images': images, 'label': label,
                  'id': id, 'gender': gender, 'age': age,
                  'lymph_count': lymph_count}
        return sample