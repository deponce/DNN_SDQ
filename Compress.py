import numpy as np
import HDQ
import SDQ
import torchvision.datasets as datasets
from torchvision import transforms
import torch
import HDQ_OptD
import HDQ_OptD_SWE
import SDQ_OptD
import matplotlib.pyplot as plt
from PIL import Image

class SDQ_transforms(torch.nn.Module):
    def __init__(self, model="NoModel", SenMap_dir="./SenMap/", colorspace=0, Q=50, q=50, J=4, a=4, b=4,
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
        self.colorspace = colorspace

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
        # self.sen_map[0] = np.loadtxt(SenMap_dir+model+"_Y.txt")
        # self.sen_map[1] = np.loadtxt(SenMap_dir+model+"_Cb.txt")
        # self.sen_map[2] = np.loadtxt(SenMap_dir+model+"_Cr.txt")

        self.sen_map[0] = np.loadtxt(SenMap_dir+"_Y.txt")
        self.sen_map[1] = np.loadtxt(SenMap_dir+"_Cb.txt")
        self.sen_map[2] = np.loadtxt(SenMap_dir+"_Cr.txt")
    def __call__(self, compressed_img):
        compressed_img = np.asarray(compressed_img)
        compressed_img = np.transpose(compressed_img, (2,0,1))
        compressed_img, BPP = SDQ.__call__(compressed_img, self.sen_map, self.model, self.colorspace, self.J, self.a, self.b, 
                                           self.Q, self.q, self.Beta_S, self.Beta_W, self.Beta_X, self.Lambda, 0.)
        compressed_img = np.round(compressed_img)    
        compressed_img = np.uint8(compressed_img) 
        compressed_img = np.transpose(compressed_img, (1,2,0))
        return {'image': compressed_img, 'BPP': BPP}


class SDQ_OptD_transforms(torch.nn.Module):
    def __init__(self, model="NoModel", SenMap_dir="./SenMap/", colorspace=0, J=4, a=4, b=4,
                        DT_Y=1, DT_C=1, d_waterlevel_Y=-1, d_waterlevel_C=-1, Qmax_Y=46, Qmax_C=46,
                        Lambda=1, Beta_S=1,Beta_W=1,Beta_X=1,):

        self.model = model
        # self.Q = Q
        # self.q = q
        self.J = J
        self.a = a
        self.b = b
        self.Lambda = Lambda
        self.Beta_S = Beta_S
        self.Beta_W = Beta_W
        self.Beta_X = Beta_X
        self.colorspace = colorspace

        self.DT_Y = DT_Y
        self.DT_C = DT_C
        self.d_waterlevel_Y = d_waterlevel_Y
        self.d_waterlevel_C = d_waterlevel_C
        self.Qmax_Y = Qmax_Y
        self.Qmax_C = Qmax_C   
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
        # self.sen_map[0] = np.loadtxt(SenMap_dir+model+"_Y.txt")
        # self.sen_map[1] = np.loadtxt(SenMap_dir+model+"_Cb.txt")
        # self.sen_map[2] = np.loadtxt(SenMap_dir+model+"_Cr.txt")

        self.sen_map[0] = np.loadtxt(SenMap_dir+"_Y.txt")
        self.sen_map[1] = np.loadtxt(SenMap_dir+"_Cb.txt")
        self.sen_map[2] = np.loadtxt(SenMap_dir+"_Cr.txt")
    def __call__(self, compressed_img):
        compressed_img = np.asarray(compressed_img)
        compressed_img = np.transpose(compressed_img, (2,0,1))
        compressed_img, BPP = SDQ_OptD.__call__(compressed_img, self.sen_map, self.model, self.colorspace, self.J, self.a, self.b, 
                                           # self.Q, self.q, 
                                           self.DT_Y, self.DT_C, self.d_waterlevel_Y, 
                                           self.d_waterlevel_C, self.Qmax_Y, self.Qmax_C,
                                           self.Beta_S, self.Beta_W, self.Beta_X, self.Lambda, 0.)
        compressed_img = np.round(compressed_img)    
        compressed_img = np.uint8(compressed_img) 
        compressed_img = np.transpose(compressed_img, (1,2,0))
        return {'image': compressed_img, 'BPP': BPP}

class HDQ_transforms(torch.nn.Module):
    def __init__(self, model="VGG11", colorspace=0, Q=50, q=50, J=4, a=4, b=4):
        self.Q = Q
        self.q = q
        self.J = J
        self.a = a
        self.b = b
        self.model = model
        self.colorspace=colorspace
    def __call__(self, sample):
        sample = np.asarray(sample)
        sample = np.transpose(sample, (2,0,1))
        # compressed_img, BPP  = sample, 0.0
        compressed_img, BPP = HDQ.__call__(sample, self.model, self.colorspace, self.J, self.a, self.b,
                                           self.Q,self.q)
        compressed_img = np.round(compressed_img)    
        compressed_img = np.uint8(compressed_img) 
        compressed_img = np.transpose(compressed_img, (1,2,0))
        return {'image': compressed_img, 'BPP': BPP}

class HDQ_OptD_transforms(torch.nn.Module):
    def __init__(self, model="NoModel", SenMap_dir="./SenMap/", colorspace=0, J=4, a=4, b=4, QF_Y=50, QF_C=50, DT_Y=1, DT_C=1, d_waterlevel_Y=-1, d_waterlevel_C=-1, Qmax_Y=46, Qmax_C=46):
        self.J = J
        self.a = a
        self.b = b
        self.QF_Y = QF_Y
        self.QF_C = QF_C
        self.model = model
        self.colorspace=colorspace
        self.DT_Y = DT_Y
        self.DT_C = DT_C
        self.d_waterlevel_Y = d_waterlevel_Y
        self.d_waterlevel_C = d_waterlevel_C
        self.Qmax_Y = Qmax_Y
        self.Qmax_C = Qmax_C        

        self.sen_map = np.ones((3,64))
        self.sen_map[0] = np.loadtxt(SenMap_dir+"_Y.txt")
        self.sen_map[1] = np.loadtxt(SenMap_dir+"_Cb.txt")
        self.sen_map[2] = np.loadtxt(SenMap_dir+"_Cr.txt")

        # print(self.sen_map)

    def __call__(self, sample):
        sample = np.asarray(sample)
        sample = np.transpose(sample, (2,0,1))
        # compressed_img, BPP  = sample, 0.0
        # OLD ---> HDQ_OptD.__call__
        compressed_img, BPP = HDQ_OptD_SWE.__call__(sample, self.sen_map, self.model, self.colorspace, self.J, self.a, self.b, self.QF_Y, self.QF_C,
                                           self.DT_Y, self.DT_C, self.d_waterlevel_Y, 
                                           self.d_waterlevel_C, self.Qmax_Y, self.Qmax_C)
        compressed_img = np.round(compressed_img)    
        compressed_img = np.uint8(compressed_img)
        compressed_img = np.transpose(compressed_img, (1,2,0))
        # plt.imshow(compressed_img)
        # plt.show()
        return {'image': compressed_img, 'BPP': BPP}
