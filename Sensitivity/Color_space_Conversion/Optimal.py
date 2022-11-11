import numpy as np
import torch
from torchvision import models, datasets, transforms
import torchvision
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from matplotlib import pyplot as plt
from utils import get_zigzag, block_dct, block_idct, rgb_to_ycbcr, ycbcr_to_rgb,\
    blockify, deblockify
from model import get_model
import argparse

from loader_sampler import *

def softmax(x):
    exp_x = torch.exp(x)
    sum_x = torch.sum(exp_x, dim=1, keepdim=True)

    return exp_x/sum_x

def log_softmax(x):
    return x - torch.logsumexp(x,dim=1, keepdim=True)

def my_CrossEntropyLoss(outputs, targets, lamda=1):
    num_examples = targets.shape[0]
    batch_size = outputs.shape[0]
    outputs = log_softmax(outputs)
    outputs = outputs[range(batch_size), targets]
    outputs = - torch.sum(outputs)/num_examples
    return outputs

def main(model = 'alexnet', Batch_size = 100, Nexample= 10000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Batch_size = Batch_size
    thr = Nexample
    model_name = model
    print("code run on", device)

    # samples_count = load_dict(root="/home/h2amer/AhmedH.Salamah/ilsvrc2012")
    transform = transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ])

    dataset = random_sampler(root="/home/h2amer/AhmedH.Salamah/ilsvrc2012", t_split='train',transform=transform)


    # transform = transforms.Compose(
    #         [transforms.ToTensor(),
    #          transforms.Resize((224, 224)),
    #          transforms.Normalize(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.])])
    # # dataset = torchvision.datasets.ImageNet(root="/home/h2amer/AhmedH.Salamah/ilsvrc2012", split='train',
    #                                         transform=transform)
    # Scale2One = transforms.Normalize(mean=[0, 0, 0], std=[255., 255., 255.])

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=True)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pretrained_model = get_model(model_name)
    # print(pretrained_model)
    # exit(0)
    pretrained_model.to(device)
    pretrained_model.eval()
    sen_list = torch.empty([3, 0])
    cnt = 0
    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)  # [0,225]
        data.requires_grad = True
        output = pretrained_model(data)
        # loss = F.nll_loss(output, target)
        # print(output)
        # print(target)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        # loss = torch.nn.NLLLoss(output, target)
        # loss = my_CrossEntropyLoss(output, target)
        loss.backward()
        grad = data.grad.reshape(Batch_size, 3, -1).transpose(1, 0)
        grad = grad.reshape(3, -1).detach().cpu()
        sen_list = torch.cat((sen_list, grad), 1)
        cnt += Batch_size
        if cnt >= thr:
            break
    sen_list = sen_list - torch.mean(sen_list, 1)[:, None]
    eigmat = torch.real(torch.linalg.eig(sen_list@sen_list.transpose(1, 0)).eigenvectors.transpose(1, 0))
    print("Optimal color space conversion matrix for", model_name, "is: ")
    for i in range(3):
        for j in range(3):
            print(eigmat[i, j].item(), " ", end="")
    print(" ")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model, Batch_size, Nexample')
    parser.add_argument('-model',type=str, default='Alexnet', help='DNN model')
    parser.add_argument('-Batch_size', type=int, default=100,help='Number of examples in one batch')
    parser.add_argument('-Nexample',type=int, default=10000, help='Number of example')
    args = parser.parse_args()
    main(**vars(args))
