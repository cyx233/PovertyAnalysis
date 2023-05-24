# from wilds import get_dataset
# from wilds.common.data_loaders import get_train_loader
import pytorch_lightning as pl
import torch.utils.data as tdata
# import webdataset as wds
import os
import torchvision.transforms as transforms
from data_loader import *

def identity(x):
    return x

class wildDM(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
#         self.wds = args.native
        self.datadir = args.datadir
        self.batch_size = args.batch_size
#         self.train_size = int(0.9 * 9797)
#         self.val_size = 9797 - self.train_size 
#         self.test_size = 3963
        self.num_workers = args.num_workers
        self.mode = args.mode
        self.country_name = args.country_name
        self.metadata_path = args.metadata_path
        self.urban_flag = args.urban_flag
        self.train_val_split_ratio = args.train_val_split_ratio
        dataset_helper = DatasetHelper(csv_file = self.metadata_path, 
                              root_dir = self.datadir)
        
        train_image_paths, val_image_paths, idx_to_class, class_weights = dataset_helper.get_image_paths(self.train_val_split_ratio)
        
        self.train_dataset = WildsDataset(train_image_paths, idx_to_class)
        self.val_dataset = WildsDataset(val_image_paths, idx_to_class)
        self.class_weights = class_weights
        
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group('wildDM')
        parser.add_argument("--mode", '-mode', type = str, default = 'country')
        parser.add_argument("--country_name", '-cntry', type = str, default = '0')
        parser.add_argument("--metadata_path", '-metapath', type = str, default = '/data/sateesh/UCSD/PovertyAnalysis/HW5/public_tables/train.csv')
        parser.add_argument("--chkpt_path", '-chkptpath', type = str, default = 'checkpoints/')
        #parser.add_argument("--native", '-nwds', action = 'store_true')
        parser.add_argument("--urban_flag", '-urban', type = bool, default = False)
        parser.add_argument("--datadir", '-dr', type = str, default = '/data/sateesh/UCSD/anon_images')
        parser.add_argument("--batch_size", '-bs', type = int, default = 2)
        parser.add_argument("--num_workers", '-nw', type = int, default = 1)
        parser.add_argument("--train_val_split_ratio", '-ratio', type = float, default = 0.8)

        return parent_parser

#     def setup(self, stage):
#         dataset_helper = DatasetHelper(csv_file = self.metadata_path, 
#                               root_dir = self.datadir, country_name = self.country_name, urban = self.urban_flag)
        
#         train_image_paths, val_image_paths, idx_to_class = dataset_helper.get_image_paths(self.train_val_split_ratio)
        
#         self.train_dataset = WildsDataset(train_image_paths, idx_to_class)
#         self.val_dataset = WildsDataset(val_image_paths, idx_to_class)


    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, 
                            num_workers = self.num_workers,
                            batch_size=self.batch_size, 
                            shuffle=False)
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, 
                            num_workers = self.num_workers,
                            batch_size=self.batch_size, 
                            shuffle=False)
        return loader




