import torchvision.datasets as datasets
from torchvision import transforms
from Compress import *
import torchvision.transforms.functional as TF
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class HDQ_loader(datasets.ImageNet):
    """docstring for loader"""
    def __init__(self, model, colorspace, root, QF_Y, QF_C, J, a, b, split="val", resize_compress=True):
        # self.transforms =  transforms.Compose([
  #                                   # transforms.Resize((256, 256)),
  #                                   transforms.Scale(256),
  #                                   transforms.CenterCrop(224),
  #                                   transforms.ToTensor(),
  #                                   # transforms.Normalize(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.]),
  #                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  #                                   # HDQ_transforms(QF_Y, QF_C, J, a, b),
  #                                   ])
        self.root = root
        self.normalize_0 = transforms.Normalize(mean=[0, 0, 0]            , std=[1/255., 1/255., 1/255.])
        self.normalize_1 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        super().__init__(self.root, split="val")
        # super().__init__(self.root)

        self.HDQ_transforms = HDQ_transforms(model, colorspace, QF_Y, QF_C, J, a, b)
        
        if resize_compress:
            self.HDQ_preprocess = self.resize_compression
        else:
            self.HDQ_preprocess = self.compression_resize
        
  #       classes (list): List of the class name tuples.
  #       class_to_idx (dict): Dict with items (class_name, class_index).
  #       wnids (list): List of the WordNet IDs.
  #       wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
  #       imgs (list): List of (image path, class_index) tuples
  #       targets (list): The class_index value for each image in the dataset
    
    def resize_compression(self, sample):
        sample = transforms.Resize(256)(sample)
        sample = transforms.CenterCrop([224, 224])(sample)
        sample = self.HDQ_transforms(sample)
        BPP = sample['BPP']
        sample = sample['image']
        # imgplot = plt.imshow(sample)
        # plt.show()
        sample = transforms.ToTensor()(sample)
        sample = self.normalize_1(sample)
        return sample, BPP

    def compression_resize(self, sample):
        sample = self.HDQ_transforms(sample)
        BPP    = sample['BPP']
        sample = sample['image']
        sample = transforms.ToPILImage()(sample)
        sample = transforms.Resize(256)(sample)
        sample = transforms.CenterCrop([224, 224])(sample)
        sample = transforms.ToTensor()(sample)
        sample = self.normalize_1(sample)
        return sample, BPP
    
    def normal(self, sample):
        BPP = 0
        sample = transforms.Resize(256)(sample)
        sample = transforms.CenterCrop([224, 224])(sample)
        sample = TF.to_tensor(sample)
        sample = self.normalize_1(sample)
        return sample, BPP
    
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        sample, BPP = self.HDQ_preprocess(sample)
        # sample, BPP = self.normal(sample)
        return sample, BPP, target

    def __len__(self):
        return len(self.imgs)

class SDQ_loader(datasets.ImageNet):
    """docstring for loader"""
    def __init__(self, model, SenMap_dir, colorspace, root, QF_Y, QF_C, J, a, b, Lambda, Beta_S, Beta_W, Beta_X, split="val",
                 resize_compress=True, eps=10, iterations=3):
        # self.transforms =  transforms.Compose([
  #                                   # transforms.Resize((256, 256)),
  #                                   transforms.Scale(256),
  #                                   transforms.CenterCrop(224),
  #                                   transforms.ToTensor(),
  #                                   # transforms.Normalize(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.]),
  #                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  #                                   # HDQ_transforms(QF_Y, QF_C, J, a, b),
  #                                   ])
        self.root = root
        self.normalize_0 = transforms.Normalize(mean=[0, 0, 0]            , std=[1/255., 1/255., 1/255.])
        self.normalize_1 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        super().__init__(self.root, split="val")

        self.SDQ_transforms = SDQ_transforms(model, SenMap_dir, colorspace, QF_Y, QF_C, J, a, b, Lambda, Beta_S, Beta_W, Beta_X, eps, iterations)
        if resize_compress:
            self.SDQ_preprocess = self.resize_compression
        else:
            self.SDQ_preprocess = self.compression_resize
        
  #       classes (list): List of the class name tuples.
  #       class_to_idx (dict): Dict with items (class_name, class_index).
  #       wnids (list): List of the WordNet IDs.
  #       wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
  #       imgs (list): List of (image path, class_index) tuples
  #       targets (list): The class_index value for each image in the dataset
    
    def resize_compression(self, sample):
        sample = transforms.Resize(256)(sample)
        sample = transforms.CenterCrop([224, 224])(sample)
        sample = self.SDQ_transforms(sample)
        BPP = sample['BPP']
        sample = sample['image']
        sample = transforms.ToTensor()(sample)
        sample = self.normalize_1(sample)
        return sample, BPP

    def compression_resize(self, sample):    
        sample = self.SDQ_transforms(sample)
        BPP    = sample['BPP']
        sample = sample['image']
        sample = transforms.ToPILImage()(sample)
        sample = transforms.Resize(256)(sample)
        sample = transforms.CenterCrop([224, 224])(sample)
        sample = transforms.ToTensor()(sample)
        sample = self.normalize_1(sample)
        return sample, BPP
    
    def normal(self, sample):
        BPP = 0
        sample = transforms.Resize(256)(sample)
        sample = transforms.CenterCrop([224, 224])(sample)
        sample = TF.to_tensor(sample)
        sample = self.normalize_1(sample)
        return sample, BPP
    
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        # print(str(index) + "\t"+self.imgs[index][0])
        sample, BPP = self.SDQ_preprocess(sample)
        # sample, BPP = self.normal(sample)
        return sample, BPP, target

    def __len__(self):
        return len(self.imgs)
