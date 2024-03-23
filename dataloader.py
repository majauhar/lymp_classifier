import os
import glob
import numpy as np
from os import path
from pathlib import Path 
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from itertools import chain

def get_age(lymph_df):
  n = lymph_df.shape[0]
  ages = np.zeros(n)
  for i in range(n):
    date_of_birth = lymph_df.DOB[i]
    if '/' in date_of_birth:
      year = date_of_birth.split('/')[-1]
    elif '-' in date_of_birth:
      year = date_of_birth.split('-')[-1]

    # We do not have information of the year of the data acquisition so we will assume it was this year. 
    # It shouldn't be relevant as it is an additive bias.  
    ages[i] = 2024 - int(year)

  lymph_df_out = lymph_df.drop(columns='DOB') 
  lymph_df_out.insert(4, 'AGE', ages)
  
  return lymph_df_out

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

class PatientDataset(Dataset):
    """This dataset samples the images and metadata for each subject"""

    def __init__(self, img_dir, data_df, transform=None):
        """
        Args:
            img_dir: (str) path to the images directory.
            data_df: (DataFrame) list of subjects used.
            transform: Optional, transformations applied to the tensor
        """
        self.img_dir = img_dir
        self.transform = transform
        self.data_df = get_age(data_df)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        """
        Args:
            idx: (int) the index of the subject/session whom data is loaded.
        Returns:
            sample: (dict) corresponding data described by the following keys:
                images: (List) List with image tensors
                label: (int) the diagnosis code
                id: (str) ID of the participant
                gender: (str) 'M' or 'F'
                age: (int) age
        """

        label = self.data_df.loc[idx, 'LABEL']
        age = self.data_df.loc[idx, 'AGE']
        gender = self.data_df.loc[idx, 'GENDER']


        id = self.data_df.loc[idx, 'ID']
        folder_name = path.join(self.img_dir, id)
        images = []
        for filename in Path(folder_name).glob('*'):
          image = read_image(str(filename))

          if self.transform:
              image = self.transform(image)
          images.append(image)

        sample = {'images': images, 'label': label,
                  'id': id, 'gender': gender, 'age': age}
        return sample
