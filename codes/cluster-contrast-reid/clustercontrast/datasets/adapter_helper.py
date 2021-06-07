import torch.utils.data as util_data
from torchvision import transforms
from PIL import Image
import numpy as np

def get_base_config_from_dataset_name(datasetname: str):
    if datasetname == 'cifar':
        return  {
            "dataset": "cifar10-1",
        }
    raise ValueError('Not found Dataset: '+datasetname)


def config_dataset(config):
    if config['dataset'] == 'cifar10':
        config["data_path"] = './data' + "/cifar10/"
    if config['dataset'] == 'imagenet':
        config["data_path"] = './data' +"/imagenet/"

    config["data"] = {
        "train_set": {"list_path": config['data_path'] + "train.txt"},
        "database": {"list_path":  config["data_path"] + "database.txt"},
        "test": {"list_path":  config["data_path"] + "test.txt"}}
    return config


def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])



def onehot_to_int(v):
    return v.index(1)
