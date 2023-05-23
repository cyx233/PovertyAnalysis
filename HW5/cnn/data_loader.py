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

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class DatasetHelper:
    
    def __init__(self, csv_file, root_dir, mode = 'country', country_name = 'angola', urban = False):
        """
        This assumes that files are stored in the following structure (as in google drive):
        root_dir:
        train/
            urban/
                0/
                    image0.npz
                    image10.npz
                        .
                        .
                        
                1/
                        .
                        .
            rural/
                0/
                    image20.npz
                    image230.npz
                        .
                        .
                1/
                        .
                        .
        test/
        
        
        note: We can easily change this part of code to follow any other folder structure
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            mode (string): only supports country mode for now, can add year too
            country_name (string): name of the country on which to train for
            urban (boolean): True for urban images, False for rural
        """
        self.labels_file = pd.read_csv(csv_file)
        self.labels_file = self.labels_file.loc[self.labels_file.urban == urban]
        if mode == 'country':
            self.labels_file = self.labels_file[self.labels_file.country == int(country_name)]
        
        self.labels_map = self.labels_file.loc[:, ['filename', 'wealthpooled', 'country', 'urban', 'label']].to_dict(orient='records')
        print('labels count:', self.labels_file.label.value_counts())
        self.root_dir = root_dir
        self.urban = urban
        self.country_name = country_name
        #self.mode = mode
        
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
        
        if self.urban:
            train_image_paths = [os.path.join(self.root_dir, 'urban', self.country_name,
                                   self.labels_map[idx]['filename']) for idx in train_image_paths]
            val_image_paths = [os.path.join(self.root_dir, 'urban', self.country_name,
                                   self.labels_map[idx]['filename']) for idx in val_image_paths]

        else:
            train_image_paths = [os.path.join(self.root_dir, 'rural', self.country_name,
                                   self.labels_map[idx]['filename']) for idx in train_image_paths]
            val_image_paths = [os.path.join(self.root_dir, 'rural', self.country_name,
                                   self.labels_map[idx]['filename']) for idx in val_image_paths]

        '''
        Remove this step (next 4 lines) once entire directory is downloaded
        '''
        
        train_image_paths = [x for x in train_image_paths if os.path.exists(x)]
        val_image_paths = [x for x in val_image_paths if os.path.exists(x)]
        print('Train Size:',len(train_image_paths))
        print('Val Size:', len(val_image_paths))
        
        idx_to_class = {x['filename']:x['label'] for x in self.labels_map}
        class_weights = compute_class_weight('balanced', [0,1], [idx_to_class[x.split('/')[-1]] for x in train_image_paths])
        class_weights = ' '.join([str(x) for x in class_weights])
        return train_image_paths, val_image_paths, idx_to_class, class_weights
    
class WildsDataset(Dataset):
    """Wilds povertyMap Dataset"""
    
    def __init__(self, image_paths, idx_to_class, transform = None, output_filenames=False):
        """
        Args:
            
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.image_paths = image_paths
        self.idx_to_class = idx_to_class
        self.transform = transform
        self.output_filenames = output_filenames
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        #Load npz file and retrieve numpy array from compressed format
        image = np.load(self.image_paths[idx])
        image = image.f.x
        
        index = self.image_paths[idx].split('/')[-1]
        label = self.idx_to_class[index]
        
        # We change wealthpooled column (a floating point number) into 3 class problem
#         if label < 0:
#             label = 0
#         elif label < 1:
#             label = 1
#         else:
#             label = 2
            

        if self.transform:
            image = self.transform(image)
        return image, label, self.image_paths[idx].split('/')[-1] if self.output_filenames else image, label
        
