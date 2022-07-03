import numpy as np
import SDQ
import torchvision.datasets as datasets
from torchvision import transforms
import torch
import HDQ
import matplotlib.pyplot as plt
from PIL import Image

class SDQ_transforms(torch.nn.Module):
    def __init__(self, model="NoModel", Q=50, q=50, J=4, a=4, b=4,
                 Lambda=1, Beta_S=1,Beta_W=1,Beta_X=1,):

        self.model = model
        self.Q = Q
        self.q = q
        self.J = J
        self.a = a
        self.b = b
        self.Lambda = Lambda
        self.Beta_S = Beta_S
        self.Beta_W = Beta_W
        self.Beta_X = Beta_X

        # print("Model: ", model)
        # print("J =", J)
        # print("a =", a)
        # print("b =", b)
        # print("QF_Y =",Q)
        # print("QF_C =",q)
        # print("Beta_S=",Beta_S)
        # print("Beta_W=",Beta_W)
        # print("Beta_X=",Beta_X)
        # print("Lambda=",Lambda)
        # exit(0)
        self.sen_map = np.ones((3,64))
        self.sen_map[0] = np.loadtxt("./SenMap/"+model+"_Y_KLT.txt")
        self.sen_map[1] = np.loadtxt("./SenMap/"+model+"_Cb_KLT.txt")
        self.sen_map[2] = np.loadtxt("./SenMap/"+model+"_Cr_KLT.txt")
    def __call__(self, img):
        img = img.detach().cpu().numpy()
        compressed_img, BPP = SDQ.__call__(img, self.sen_map, self.model, self.J, self.a, self.b, 
                                           self.Q, self.q, self.Beta_S, self.Beta_W, self.Beta_X, self.Lambda, 0.)
        compressed_img = torch.tensor(compressed_img)
        return{'image': compressed_img, 'BPP': BPP}

class SDQ_transforms_raw(torch.nn.Module):
    def __init__(self, model="NoModel", Q=50, q=50, J=4, a=4, b=4,
                 Lambda=1, Beta_S=1,Beta_W=1,Beta_X=1,):

        self.model = model
        self.Q = Q
        self.q = q
        self.J = J
        self.a = a
        self.b = b
        self.Lambda = Lambda
        self.Beta_S = Beta_S
        self.Beta_W = Beta_W
        self.Beta_X = Beta_X
        self.sen_map = np.ones((3,64))
        self.sen_map[0] = np.loadtxt("./SenMap/"+model+"_Y_KLT.txt")
        self.sen_map[1] = np.loadtxt("./SenMap/"+model+"_Cb_KLT.txt")
        self.sen_map[2] = np.loadtxt("./SenMap/"+model+"_Cr_KLT.txt")
    def __call__(self, img):
        img = img.detach().cpu().numpy()
        compressed_img, BPP = SDQ.__call__(img, self.sen_map, self.model, self.J, self.a, self.b, 
                                           self.Q, self.q, self.Beta_S, self.Beta_W, self.Beta_X, self.Lambda, 0.)
        # compressed_img = torch.tensor(compressed_img)
        return{'image': compressed_img, 'BPP': BPP}

class HDQ_transforms(torch.nn.Module):
    def __init__(self, model="VGG11", Q=50, q=50, J=4, a=4, b=4):
        self.Q = Q
        self.q = q
        self.J = J
        self.a = a
        self.b = b
        self.model = model
    def __call__(self, img):
        img = img.detach().cpu().numpy()
        compressed_img, BPP = HDQ.__call__(img, self.model, self.J, self.a, self.b,
                                           self.Q,self.q)
        compressed_img = torch.tensor(compressed_img) #--> /255
        return {'image': compressed_img, 'BPP': BPP}

class HDQ_transforms_raw(torch.nn.Module):
    def __init__(self, model="VGG11", Q=50, q=50, J=4, a=4, b=4):
        self.Q = Q
        self.q = q
        self.J = J
        self.a = a
        self.b = b
        self.model = model
    def __call__(self, img):
        img = img.detach().cpu().numpy()
        compressed_img, BPP = HDQ.__call__(img, self.model, self.J, self.a, self.b,
                                           self.Q,self.q)
        # compressed_img = torch.tensor(compressed_img)
        return {'image': compressed_img, 'BPP': BPP}
