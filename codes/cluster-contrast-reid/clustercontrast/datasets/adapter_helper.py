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
    prefix = '/home/chris/research_dataset/hash_dataset'
    if config['dataset'] == 'cifar10':
        config["data_path"] = prefix + "/cifar10/"
    if config['dataset'] == 'imagenet':
        config["data_path"] = prefix+"/imagenet/"

    config["data"] = {
        "train_set": {"list_path": config['dataset'] + "train.txt"},
        "database": {"list_path":  config["dataset"] + "database.txt"},
        "test": {"list_path":  config["dataset"] + "test.txt"}}
    return config

class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


class ImageListWOTransform(object):
    def __init__(self, data_path, image_list):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]

    def __getitem__(self, index):
        path, target = self.imgs[index]
        # img = Image.open(path).convert('RGB')
        return path, target, index

    def __len__(self):
        return len(self.imgs)


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


def get_data(config):

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set", "test", "database"]:
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set))
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]["batch_size"],
                                                      shuffle=True, num_workers=4)

    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])


def onehot_to_int(v):
    return v.index(1)

def get_data_without_transform(config):
    dsets = {}
    data_config = config["data"]

    for data_set in ["train_set", "test"]:
        dsets[data_set] = ImageListWOTransform(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines())
        print(data_set, len(dsets[data_set]))

    return dsets["train_set"], dsets["test"],  \
           len(dsets["train_set"]), len(dsets["test"])