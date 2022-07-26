import numpy as np
import tqdm
import torchvision.datasets as datasets
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from torchvision import models
# from PIL import Image
from utils import load_model, print_file, print_exp_details_SDQ
from Compress import SDQ_transforms
from Utils.loader import SDQ_loader 
import argparse
import random
import warnings


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
    Batch_size = 1
    model = args.Model
    J = args.J
    a = args.a
    b = args.b
    QF_Y = args.QF_Y
    QF_C = args.QF_C
    Beta_S = args.Beta_S
    Beta_W = args.Beta_W
    Beta_X = args.Beta_X
    Lmbd = args.L
    resize_compress = args.resize_compress
    iterations =  args.iterations
    eps = 10
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # pretrained_model = load_model(model)
    print(device)
    print_exp_details_SDQ(args)
    print("Model: ", model)
    print("J =", J)
    print("a =", a)
    print("b =", b)
    print("QF_Y =",QF_Y)
    print("QF_C =",QF_C)
    print("Beta_S=",Beta_S)
    print("Beta_W=",Beta_W)
    print("Beta_X=",Beta_X)
    print("Lambda=",Lmbd)
    print("iterations=",iterations)
    pretrained_model = []
    pretrained_name = [
                        "VGG11", 
                        "Resnet18", 
                        "Squeezenet", "Alexnet"
    ]

    for nm in pretrained_name: 
        pretrained_model_buff = load_model(nm)
        _ = pretrained_model_buff.to(device)
        pretrained_model.append(pretrained_model_buff)
    
    # transform = transforms.Compose([
    #                                 transforms.Scale(256),
    #                                 transforms.CenterCrop(224),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.]),
    #                                 # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #                                 SDQ_transforms(model, QF_Y, QF_C, J, a, b, Lmbd, Beta_S, Beta_W, Beta_X)
    #                                 ])
    # dataset = datasets.ImageNet(root="/home/h2amer/AhmedH.Salamah/ilsvrc2012", split='val', transform=transform)
    
    # The Colorspace will be always 0 .... as it is the vanila SDQ with YUV only that will be similar for all networks.
    model = "VGG11"
    args.colorspace = 0
    args.sens_dir="./SenMap_All/NoModel/VGG11"
    dataset = SDQ_loader(model, SenMap_dir=args.SenMap_dir, root=args.root, colorspace=args.colorspace,  QF_Y=QF_Y, QF_C=QF_C, J=J, a=a, b=b, Lambda=Lmbd,
                                Beta_S=Beta_S, Beta_W=Beta_W, Beta_X=Beta_X, split="val", resize_compress=resize_compress)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    num_correct = [0] * len(pretrained_model)
    num_tests = 0
    BPP = 0
    cnt = 0
    for dt in tqdm.tqdm(test_loader):
        image, image_BPP, labels = dt
        labels = labels.to(device)
        image = image.to(device)
        if torch.sum(image_BPP) < 0:
           break
        BPP+=torch.sum(image_BPP)
        num_tests += len(labels)
        for idx , model in enumerate(pretrained_model): 
            pred = pretrained_model[idx](image)
            num_correct[idx] += (pred.argmax(1) == labels).sum().item()
            if (cnt+1) %500 ==0:
                l0 = pretrained_name[idx] + " --> " + str(cnt) + "\n"
                l1 = str(num_correct[idx]/num_tests) + " = " + str(num_correct[idx]) + " / "+ str(num_tests) + "\n"
                l2 = str(BPP.numpy()/num_tests) + "\n"
                # l2 = ""
                l = l0 + l1 + l2
                print_file(l, args.output_txt)
        cnt += 1

    for idx , model in enumerate(pretrained_model): 
        # l0 = "#"* 30 + "\n"
        # l0 = l0 + pretrained_name[idx] + "\n"
        # l1 = str(num_correct[idx]/num_tests) + " = " + str(num_correct[idx]) + " / "+ str(num_tests) + "\n"
        # l2 = str(BPP.numpy()/num_tests) + "\n"
        # l = l0 + l1 + l2
        # print_file(l, args.output_txt)
        l0 = "*"* 30 + "\n"
        l0 = l0 + pretrained_name[idx] + "\n"
        l1 = str((num_correct[idx]/num_tests)*100) + "\t" + str(BPP.numpy()/num_tests) + "\n"
        l = l0 + l1
        print_file(l, args.output_txt)

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description="SDQ")
    parser.add_argument('--Model', type=str, default="Alexnet", help='Subsampling b')
    parser.add_argument('--J', type=int, default=4, help='Subsampling J')
    parser.add_argument('--a', type=int, default=4, help='Subsampling a')
    parser.add_argument('--b', type=int, default=4, help='Subsampling b')
    parser.add_argument('--QF_Y', type=int, default=50, help='Quality factor of Y channel')
    parser.add_argument('--QF_C', type=int, default=50, help='Quality factor of Cb & Cr channel')
    parser.add_argument('--Beta_S', type=float, default=100, help='Subsampling b')
    parser.add_argument('--Beta_W', type=float, default=100, help='Subsampling b')
    parser.add_argument('--Beta_X', type=float, default=100, help='Subsampling b')
    parser.add_argument('--L', type=float, default=1, help='Subsampling b')
    parser.add_argument('--output_txt', type=str, help='output txt file')
    parser.add_argument('--device', type=str, default="cuda:1", help='cpu or cuda:1')
    parser.add_argument('-resize_compress', action='store_true', help='For Resize --> Compress set True')
    parser.add_argument('--root', type=str, default="/home/h2amer/AhmedH.Salamah/ilsvrc2012", 
                            help='root to ImageNet Directory')
    parser.add_argument('--SenMap_dir', type=str, default="./SenMap/", 
                            help='Senstivity Directory')
    parser.add_argument('--colorspace', type=int, default=0, help='ColorSpace 0:YUV 1:SWX')
    parser.add_argument('--iterations', type=int, default=3, help='Numebr of iterations to run SDQ')
    args = parser.parse_args()
    main(args)
