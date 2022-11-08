import numpy as np
import tqdm
import torchvision.datasets as datasets
from torchvision import transforms
import torch
# import matplotlib.pyplot as plt
from torchvision import models
# from PIL import Image
from utils import *
from Utils.loader import HDQ_loader 
import argparse
import random
import warnings
import pickle

num_workers=24

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**num_workers
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

def warn(*args, **kwargs):
    pass
warnings.warn = warn


def main(args):
    fileFormat = args.output_txt
    data_all = {}
    sens = args.SenMap_dir.split("/")[2]
    data_file_name =  args.Model+"_" + sens
    const = 1
    if sens == "NoModel":
        const = 10
    # dy_list = []
    # dy_list = [0.01]
    # dc_list = [0.01]

    qy_list = [95]
    qc_list = [95]

    # qy_list = np.arange(100,80,-1)
    # qc_list = np.arange(100,80,-1)

    # dy_list.extend(np.arange(0.005, 0.01, 0.001))
    # dy_list.extend(np.arange(0.01, 0.11, 0.01))
    # d_waterlevel_Y=[0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04]
    # d_waterlevel_C=[0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20]
    tmp = 0
    for i in range(len(qy_list)):
        # args.d_waterlevel_Y = dy_list[i] * const
        # args.d_waterlevel_C = dc_list[i] * const
        args.QF_Y = qy_list[i]
        args.QF_C = qc_list[i]
        args.Qmax_Y = 46
        args.Qmax_C = 46
        # for ratio in [3/4, 1 , 5/4, 6/4, 7/4, 8/4]:
        # for ratio in [1/4, 2/4, 3/4, 1 , 5/4, 6/4, 7/4, 8/4, 9/4, 10/4 , 11/4, 12/4, 13/4, 14/4, 15/4, 16/4]:
            # for ratio in [1]:
        # for args.Qmax_Y in range(1, 62, 3):
        #     for args.Qmax_C in range(args.Qmax_Y, 62, 3):
            # for ratio in [1]:  
                # args.d_waterlevel_C = ratio * args.d_waterlevel_Y
                # args.d_waterlevel_C = d_waterlevel_C[i] * const
                # max_q_c = np.ceil(255/Q)
                # for ratio in np.arange(1, max_q_c+1):
                #     args.Qmax_C = int(min(ratio * Q , 255))
        args.output_txt = fileFormat%(args.QF_Y, args.QF_C, args.Qmax_Y, args.Qmax_C)
        # print(args.output_txt)
        BPP, Acc, Qmax_flag= running_func(args)
        # BPP , Acc = 0 , 0
        key = str(args.QF_Y) + "_" + str(args.QF_C) + "_" + str(args.Qmax_Y) + "_" + str(args.Qmax_C) + "_" +str(Qmax_flag)
        # data_all[key] = [BPP, Acc]
        write_live("./RESULTS_new_senmap_SWE/"+data_file_name, key, [BPP, Acc])
        if (abs(tmp - BPP ) < 1e-4) or Qmax_flag:
            break
        tmp = BPP

    #         break
    #     break
    # break

    # with open("./RESULTS/"+data_file_name +'.pkl', 'wb') as handle:
    #     pickle.dump(data_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
   


def main_low_rate(args):
    fileFormat = args.output_txt
    data_all = {}
    sens = args.SenMap_dir.split("/")[2]
    data_file_name =  args.Model+"_" + sens
    const = 1
    if sens == "NoModel":
        const = 10
    # dy_list = np.concatenate((np.arange(0.06, 0.1, 0.005), np.arange(0.1, 0.21, 0.01)))
    # dc_list = np.concatenate((np.arange(0.05, 0.1, 0.01), np.arange(0.1, 0.21, 0.01)))

    # dy_list = np.arange(0.02, 0.2, 0.025)
    # dc_list = np.arange(0.02, 0.2, 0.05)

    dy_list = np.arange(0.52, 1.02, 0.05)
    dc_list = np.arange(0.52, 1.02, 0.05)
    tmp = 0
    for idx1, i in enumerate(dy_list):
        for idx2, j in enumerate(dc_list):
            args.d_waterlevel_Y = dy_list[idx1] * const
            args.d_waterlevel_C = dc_list[idx2] * const

            for args.Qmax_Y in range(25, 76, 5):
                for args.Qmax_C in range(args.Qmax_Y, 76, 5):
                    args.output_txt = fileFormat%(args.d_waterlevel_Y, args.d_waterlevel_C, args.Qmax_Y, args.Qmax_C)
                    # print(args.output_txt)
                    BPP, Acc, Qmax_flag= running_func(args)
                    # BPP , Acc = 0 , 0
                    key = str(args.d_waterlevel_Y) + "_" + str(args.d_waterlevel_C) + "_" + str(args.Qmax_Y) + "_" + str(args.Qmax_C) + "_" +str(Qmax_flag)
                    # data_all[key] = [BPP, Acc]
                    write_live("./RESULTS_new_senmap_low_rate/"+data_file_name, key, [BPP, Acc])



    # with open("./RESULTS/"+data_file_name +'.pkl', 'wb') as handle:
    #     pickle.dump(data_all, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    
def write_live(filename, key, vec):
    f = open(filename +'.txt', "+a")
    f.write(key + "\t")
    for x in vec:
        f.write(str(x)+ "\t")
    f.write("\n")
    f.close()



def running_func(args):
    Batch_size = 10
    model = args.Model
    J = args.J
    a = args.a
    b = args.b
    Qmax_Y = args.Qmax_Y
    Qmax_C = args.Qmax_C
    QF_Y = args.QF_Y
    QF_C = args.QF_C
    DT_Y = args.DT_Y
    DT_C = args.DT_C
    d_waterlevel_Y = args.d_waterlevel_Y
    d_waterlevel_C = args.d_waterlevel_C
    resize_compress = args.resize_compress
    OptD = args.OptD
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # print_exp_details(args)
    print("Model: ", model)
    print("Colorspace: ", args.colorspace)
    # print("J =", J)
    # print("a =", a)
    # print("b =", b)
    print("QF_Y:", QF_Y)
    print("QF_C:", QF_C)
    # print("DT_Y:", DT_Y)
    # print("DT_C:", DT_C)
    # print("d_waterlevel_Y: ",d_waterlevel_Y)
    # print("d_waterlevel_C: ",d_waterlevel_C)
    print("Qmax_Y =",Qmax_Y)
    print("Qmax_C =",Qmax_C)
    print("OptD enables =",OptD)


    # pretrained_model = models.vgg11(pretrained=True)
    # pretrained_model = models.resnet18(pretrained=True)
    # pretrained_model = models.squeezenet1_0(pretrained=True)
    # pretrained_model = models.alexnet(pretrained=True)
    pretrained_model = load_model(model) 
    _ = pretrained_model.to(device)
    # transform = transforms.Compose([
    #                                 HDQ_transforms(QF_Y, QF_C, J, a, b),
    #                                 transforms.Scale(256),
    #                                 transforms.CenterCrop(224),
    #                                 transforms.ToTensor(),
    #                                 # transforms.ToTensor(),
    #                                 transforms.Normalize(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.]),
    #                                 ])
    # dataset = datasets.ImageNet(root="/home/h2amer/AhmedH.Salamah/ilsvrc2012", split='val', transform=transform)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = HDQ_loader(   model=model, SenMap_dir=args.SenMap_dir, root=args.root, QF_Y=QF_Y, QF_C=QF_C, 
                            colorspace=args.colorspace, J=J, a=a, b=b,
                            DT_Y=DT_Y, DT_C=DT_C, d_waterlevel_Y=d_waterlevel_Y, d_waterlevel_C=d_waterlevel_C, QMAX_Y=Qmax_Y, QMAX_C=Qmax_C,
                            split="val", resize_compress=resize_compress, OptD=args.OptD)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    num_correct = 0
    num_tests = 0
    BPP = 0
    cnt = 0
    count_Qmax = 0
    Qmax_flag = False
    loss = 0
    for dt in tqdm.tqdm(test_loader):
        image, image_BPP, labels = dt
        count_Qmax += torch.sum(image_BPP < 0)
        image_BPP = torch.abs(image_BPP)
        # exit(0)
        labels = labels.to(device)
        image = image.to(device)
        BPP+=torch.sum(image_BPP)
        pred = pretrained_model(image)
        loss += float(torch.nn.CrossEntropyLoss()(pred, labels))
        num_correct += (pred.argmax(1) == labels).sum().item()
        num_tests += len(labels)
        
        if (cnt+1) %100 ==0:
            l0 = "--> " + str(cnt) + "\n"
            l1 = str(num_correct/num_tests) + " = " + str(num_correct) + " / "+ str(num_tests) + "\n"
            l2 = str(BPP.numpy()/num_tests) + "\n"
            l = l0 + l1 + l2
            print_file(l, args.output_txt)
        cnt += 1


    l0 = "#"* 30 + "\n"
    l1 = str(num_correct/num_tests) + " = " + str(num_correct) + " / "+ str(num_tests) + "\n"
    l2 = str(BPP.numpy()/num_tests) + "\n"
    l = l0 + l1 + l2
    print_file(l, args.output_txt)
    l0 = "*"* 30 + "\n"
    l1 = str((num_correct/num_tests)*100) + "\t" + str(BPP.numpy()/num_tests) + "\n"
    l = l0 + l1
    print_file(l, args.output_txt)
    l = str(loss/num_tests) + "\n"
    print_file("average loss = "+l, args.output_txt)
    
    if (count_Qmax == len(dataset)): Qmax_flag = True
    
    return (BPP.numpy()/num_tests), ((num_correct/num_tests)*100) , Qmax_flag


if '__main__' == __name__:
    parser = argparse.ArgumentParser(description="HDQ")
    parser.add_argument('--Model', type=str, default="Alexnet", help='Subsampling b')
    parser.add_argument('--J', type=int, default=4, help='Subsampling J')
    parser.add_argument('--a', type=int, default=4, help='Subsampling a')
    parser.add_argument('--b', type=int, default=4, help='Subsampling b')
    parser.add_argument('--Qmax_Y', type=int, default=46, help='Maximum Quantization Step Y Channel')
    parser.add_argument('--Qmax_C', type=int, default=46, help='Maximum Quantization Step C Channel')
    parser.add_argument('--d_waterlevel_Y', type=float, default=-1, help='Waterfilling level on Y channel')
    parser.add_argument('--d_waterlevel_C', type=float, default=-1, help='Waterfilling level on C channel')
    parser.add_argument('--DT_Y', type=float, default=1, help='Target Distortion on Y channel')
    parser.add_argument('--DT_C', type=float, default=1, help='Target Distortion on C channel')
    parser.add_argument('--QF_Y', type=float, default=100, help='QF of Y channel')
    parser.add_argument('--QF_C', type=float, default=100, help='QF of C channel')
    parser.add_argument('-resize_compress', action='store_true', help='For Resize --> Compress set True')
    parser.add_argument('--output_txt', type=str, help='output txt file')
    parser.add_argument('--device', type=str, default="cuda:0", help='cpu or cuda:0')
    parser.add_argument('--root', type=str, default="/home/h2amer/AhmedH.Salamah/ilsvrc2012", 
                            help='root to ImageNet Driectory')
    parser.add_argument('--SenMap_dir', type=str, default="./SenMap/", 
                            help='Senstivity Directory')
    parser.add_argument('--colorspace', type=int, default=0, help='ColorSpace 0:YUV 1:SWX')
    parser.add_argument('--OptD', type=bool, default=False, help='OptD initialization for Quantization Table')
    args = parser.parse_args()
    main(args)
    # main_low_rate(args)