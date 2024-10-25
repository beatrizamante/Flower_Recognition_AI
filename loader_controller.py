import torch
from torch import utils 
from torchvision import datasets, transforms

class data_load:
    def __init__(self, data_dir):
        self.image_datasets = None
        self.data_loaders = None
        self.data_dir = data_dir
        self.__init_data_load()

    def __init_data_load(self):
        train_transforms = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transforms = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        valid_transforms = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.image_datasets = {
            'train': datasets.ImageFolder(self.data_dir + '/train', transform=train_transforms),
            'test': datasets.ImageFolder(self.data_dir + '/test', transform=test_transforms),
            'valid': datasets.ImageFolder(self.data_dir + '/valid', transform=valid_transforms)
        }

        self.data_loaders = {
            'train': utils.data.DataLoader(self.image_datasets['train'], batch_size=164, shuffle=True),
            'test': utils.data.DataLoader(self.image_datasets['test'], batch_size=164, shuffle=True),
            'valid': utils.data.DataLoader(self.image_datasets['valid'], batch_size=164, shuffle=True)
        }
        

