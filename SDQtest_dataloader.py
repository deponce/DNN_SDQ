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

torch.manual_seed(0)

def warn(*args, **kwargs):
    pass
import warnings
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
    eps = 10
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    pretrained_model, _ = load_model(model)
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
    # pretrained_model = models.vgg11(pretrained=True)
    # pretrained_model = models.resnet18(pretrained=True)
    # pretrained_model = models.squeezenet1_0(pretrained=True)
    # pretrained_model = models.alexnet(pretrained=True)
    # pretrained_model, model = load_model(model) 
    _ = pretrained_model.to(device)
    # transform = transforms.Compose([
    #                                 transforms.Scale(256),
    #                                 transforms.CenterCrop(224),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize(mean=[0, 0, 0], std=[1/255., 1/255., 1/255.]),
    #                                 # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #                                 SDQ_transforms(model, QF_Y, QF_C, J, a, b, Lmbd, Beta_S, Beta_W, Beta_X)
    #                                 ])
    # dataset = datasets.ImageNet(root="/home/h2amer/AhmedH.Salamah/ilsvrc2012", split='val', transform=transform)
    

    dataset = SDQ_loader(model, SenMap_dir=args.SenMap_dir, root=args.root, colorspace=args.colorspace,  QF_Y=QF_Y, QF_C=QF_C, J=J, a=a, b=b, Lambda=Lmbd,
                                Beta_S=Beta_S, Beta_W=Beta_W, Beta_X=Beta_X, split="val", resize_compress=resize_compress)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=True, num_workers=32)
    num_correct = 0
    num_tests = 0
    BPP = 0
    cnt = 0
    for dt in tqdm.tqdm(test_loader):
        image, image_BPP, labels = dt
        labels = labels.to(device)
        image = image.to(device)
        BPP+=torch.sum(image_BPP)
        pred = pretrained_model(image)
        num_correct += (pred.argmax(1) == labels).sum().item()
        num_tests += len(labels)
        if (cnt+1) %1000 ==0:
            l0 = "--> " + str(cnt) + "\n"
            l1 = str(num_correct/num_tests) + " = " + str(num_correct) + " / "+ str(num_tests) + "\n"
            l2 = str(BPP.numpy()/num_tests) + "\n"
            # l2 = ""
            l = l0 + l1 + l2
            print_file(l, args.output_txt)
        cnt += 1

    l0 = "#"* 30 + "\n"
    l1 = str(num_correct/num_tests) + " = " + str(num_correct) + " / "+ str(num_tests) + "\n"
    l2 = str(BPP.numpy()/num_tests) + "\n"
    l = l0 + l1 + l2
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
    args = parser.parse_args()
    main(args)
