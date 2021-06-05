from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import re
import os.path as osp
import numpy as np

from ..utils.data import BaseImageDataset
from .adapter_helper import get_base_config_from_dataset_name, get_data_without_transform, config_dataset

class Cifar101(BaseImageDataset):
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
        super(Cifar101, self).__init__()
        config = get_base_config_from_dataset_name('cifar')
        config = config_dataset(config)
        train_imagelist, test_imagelist, num_train, num_test = get_data_without_transform(config)
        train = self.process_imagelist(train_imagelist, relabel=True)
        test = self.process_dir(test_imagelist, relabel=False)

        self.train = train
        self.query = []
        self.gallery = test

    def process_imagelist(self, imagelist, relabel=False):
        datasets = []
        for img, target, index in imagelist:
            datasets.append((img, np.argmax(target), 0))
        return datasets

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 776  # pid == 0 means background
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
