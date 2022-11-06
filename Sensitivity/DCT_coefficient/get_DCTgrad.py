import numpy as np
import torch
from torchvision import transforms
import torchvision
from tqdm import tqdm
import torch.nn.functional as F
from utils import *
import os
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.nn import DataParallel
# import torch.distributed as dist
from loader_sampler import *

# os.environ["CUDA_VISIBLE_DEVICES"]='1'

from model import get_model
import argparse

NUM_SAMPLES_PER_CLASS=10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# samples_count = {}
# # a simple custom collate function, just to show the idea
# def my_collate(batch):
#     data = [item[0] for item in batch]
#     target = [item[1] for item in batch]
#     if target not in samples_count:
#         samples_count[target] = 0
#     if samples_count[target] > NUM_SAMPLES_PER_CLASS:
#         next
#     samples_count[target] += 1
#     print(samples_count)
#     return [data, target]


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
    Batch_size = Batch_size
    thr = Nexample
    model_name = model

    # main_dir = "/home/h2amer/AhmedH.Salamah/workspace_pc15/JPEG_lin/DNN_SDQ/Sensitivity/DCT_coefficient"
    main_dir = "/home/h2amer/work/workspace/JPEG_SDQ/DNN_SDQ/Sensitivity/DCT_coefficient"
    print("code run on", device)

    # Trans = [transforms.ToTensor(),
    #          transforms.Resize((256, 256)),
    #          transforms.CenterCrop(224),
    #          transforms.Normalize(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.])]
    # transform = transforms.Compose(Trans)

    # transform = transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #             transforms.Resize((224,224)),
    #             transforms.Normalize(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.])
    #         ]
    #     )

    transform = transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ])

    # dataset = random_sampler(root="/home/h2amer/AhmedH.Salamah/ilsvrc2012", t_split='train',transform=transform)
    dataset = random_sampler(root="/home/h2amer/work/workspace/ML_TS", t_split='training_original',transform=transform)

    # dataset = torchvision.datasets.ImageNet(root="~/project/data", split='train',
    #                                         transform=transform)
    
    Scale2One = transforms.Normalize(mean=[0, 0, 0], std=[255., 255., 255.])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=True, num_workers=16)
                        # , collate_fn=my_collate)

    pretrained_model = get_model(model_name, device)
    _ = pretrained_model.to(device)

    
    pretrained_model.eval()
    Y_sen_list = np.empty([0, 8, 8])
    Cb_sen_list = np.empty([0, 8, 8])
    Cr_sen_list = np.empty([0, 8, 8])
    samples_count = {}
    for i in range(0,1000):
        samples_count[i] = 0
    for data, target in tqdm(test_loader):
        # for idx, target_ in enumerate(target.cpu().numpy()): 
        #     if samples_count[target_] > NUM_SAMPLES_PER_CLASS:
        #     # if samples_count[target_] >= 1:
        #         target = torch.cat((target[:idx], target[idx+1:]), axis = 0)
        #         data = torch.cat((data[:idx,:,:,:], data[idx+1:,:,:,:]), axis = 0)
        #     samples_count[target_] += 1
        data, target = data.to(device), target.to(device)  # [0,225]
        # data = normalize(Scale2One(data))
        img_shape = data.shape[-2:]
        ycbcr_data = rgb_to_ycbcr(data)
        input_DCT_block_batch = block_dct(blockify(ycbcr_data, 8))
        input_DCT_block_batch.requires_grad = True
        recoverd_img = deblockify(block_idct(input_DCT_block_batch), (img_shape[0], img_shape[1]))  # [-128, 127]
        recoverd_RGB_img = ycbcr_to_rgb(recoverd_img)
        # norm_img1 = Scale2One(recoverd_RGB_img)  # [0,1]
        # breakpoint()
        # norm_img = normalize(norm_img1)
        output = pretrained_model(recoverd_RGB_img)
        # loss = my_CrossEntropyLoss(output, target)
        loss = torch.nn.CrossEntropyLoss()(output, target)

        # FIX ME
        # loss = F.nll_loss(output, target)
        pretrained_model.zero_grad()
        loss.backward()
        data_grad = torch.mean(torch.abs(input_DCT_block_batch.grad), dim = 2).transpose(1,0).detach().cpu().numpy()
        # breakpoint()
        Y_sen_list = np.concatenate((Y_sen_list, data_grad[0].reshape(-1, 8, 8)))
        Cb_sen_list = np.concatenate((Cb_sen_list, data_grad[1].reshape(-1, 8, 8)))
        Cr_sen_list = np.concatenate((Cr_sen_list, data_grad[2].reshape(-1, 8, 8)))
        if Y_sen_list.shape[0] >= thr:
            break
    np.save(main_dir + "/grad/Y_sen_list_" + model_name + ".npy",Y_sen_list)
    np.save(main_dir + "/grad/Cb_sen_list_" + model_name + ".npy", Cb_sen_list)
    np.save(main_dir + "/grad/Cr_sen_list_" + model_name + ".npy", Cr_sen_list)
    print(Y_sen_list.shape[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model, Batch_size, Nexample, resize')
    parser.add_argument('-model',type=str, default='alexnet', help='DNN model')
    parser.add_argument('-Batch_size', type=int, default=100,help='Number of examples in one batch')
    parser.add_argument('-Nexample',type=int, default=10000, help='Number of example')
    # parser.set_defaults(feature=True)
    args = parser.parse_args()
    main(**vars(args))
