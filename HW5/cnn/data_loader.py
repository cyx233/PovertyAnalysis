from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import random
from sklearn.utils.class_weight import compute_class_weight
random.seed(77)
import pdb
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class DatasetHelper:
    
    def __init__(self, csv_file, root_dir):
        """
      
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
         """
        self.labels_file = pd.read_csv(csv_file)
        
        self.labels_map = self.labels_file.loc[:, ['filename', 'wealthpooled', 'country', 'urban', 'label']].to_dict(orient='records')
        print('labels count:', self.labels_file.label.value_counts())
        self.root_dir = root_dir
        
    def get_image_paths(self, train_val_split_ratio = 0.8):
        '''
            This function returns a list of train and val paths.
            args:
            train_val_split_ratio: train and val split ratio
        '''
        indices = [i for i in range(len(self.labels_map))]
        random.shuffle(indices)
        train_image_paths = indices[:int(train_val_split_ratio*len(indices))]
        val_image_paths = indices[int(train_val_split_ratio*len(indices)):]
        
        print('Train Size:',len(train_image_paths))
        print('Val Size:', len(val_image_paths))
        
        train_image_paths = [os.path.join(self.root_dir,
                                self.labels_map[idx]['filename']) for idx in train_image_paths]
        val_image_paths = [os.path.join(self.root_dir, 
                                self.labels_map[idx]['filename']) for idx in val_image_paths]

        '''
        Remove this step (next 4 lines) once entire directory is downloaded
        '''
        
        train_image_paths = [x for x in train_image_paths if os.path.exists(x)]
        val_image_paths = [x for x in val_image_paths if os.path.exists(x)]
        print('Train Size:',len(train_image_paths))
        print('Val Size:', len(val_image_paths))
        
        idx_to_class = {x['filename']:x['label'] for x in self.labels_map}
        class_weights = compute_class_weight(class_weight='balanced', classes = [0,1], y = [idx_to_class[x.split('/')[-1]] for x in train_image_paths])
        class_weights = ' '.join([str(x) for x in class_weights])
        return train_image_paths, val_image_paths, idx_to_class, class_weights
    

class TestDatasetHelper:
    
    def __init__(self, csv_file, root_dir):
        """
      
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
         """
        self.labels_file = pd.read_csv(csv_file)
        
        self.labels_map = self.labels_file.loc[:, ['filename', 'country']].to_dict(orient='records')
      
        self.root_dir = root_dir
        
    def get_image_paths(self):
        '''
            This function returns a list of train and val paths.
            args:
            train_val_split_ratio: train and val split ratio
        '''
        indices = [i for i in range(len(self.labels_map))]
     #   pdb.set_trace()    
    
        test_paths = [os.path.join(self.root_dir,
                                self.labels_map[idx]['filename']) for idx in indices]
        
        test_image_paths = [x for x in test_paths if os.path.exists(x)]
        
        print('Test Size:',len(test_image_paths))
        
        
        return test_image_paths
    
class WildsDataset(Dataset):
    """Wilds povertyMap Dataset"""
    
    def __init__(self, image_paths, idx_to_class = None, transform = None, output_filenames=False, mode="train"):
        """
        Args:
            
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.image_paths = image_paths
        self.idx_to_class = idx_to_class
        self.transform = transform
        self.output_filenames = output_filenames
        self.mode = mode
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        #Load npz file and retrieve numpy array from compressed format
        image = np.load(self.image_paths[idx])
        image = image.f.x
        
        index = self.image_paths[idx].split('/')[-1]
        if self.mode == 'train':
            label = self.idx_to_class[index]
        else:
            label = -1
        if self.transform:
            image = self.transform(image)
        
        if self.output_filenames:

            return image, label, self.image_paths[idx].split('/')[-1]
        
        return image, label



if __name__ == '__main__':

    helper = DatasetHelper("../public_tables/train.csv", "/data/sateesh/UCSD/anoon_images")

