from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import re
import os.path as osp
import numpy as np

from ..utils.data import BaseImageDataset
from .adapter_helper import config_dataset

class Imagenet(BaseImageDataset):
    """
    VeRi
    Reference:
    Liu, X., Liu, W., Ma, H., Fu, H.: Large-scale vehicle re-identification in urban surveillance videos. In: IEEE   %
    International Conference on Multimedia and Expo. (2016) accepted.
    Dataset statistics:
    # identities: 776 vehicles(576 for training and 200 for testing)
    # images: 37778 (train) + 11579 (query)
    """

    def __init__(self, root, verbose=True, **kwargs):
        super(Imagenet, self).__init__()
        config = {"dataset": "imagenet"}
        config = config_dataset(config)
         
        self.train = self.get_data_items(config['data_path'],config['data']['train_set']['list_path'])
        self.query = self.get_data_items(config['data_path'],config['data']['test']['list_path'])
        self.gallery = self.get_data_items(config['data_path'],config['data']['database']['list_path'])
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


    
    def get_data_items(self,data_path, list_path):
        dataset_items = []
        image_lists = open(list_path).readlines()
        
        for line in image_lists:
            image_path = data_path + line.split()[0]
            target = np.array([int(la) for la in line.split()[1:]])
            target = np.argmax(target)
            dataset_items.append((image_path, target, 0))
        
        return dataset_items
