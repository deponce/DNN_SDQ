import numpy as np
import tqdm
import torchvision.datasets as datasets
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from torchvision import models
# from PIL import Image
from utils import load_model, print_file, print_exp_details
from Compress import HDQ_transforms
from Utils.loader import HDQ_loader 
import argparse

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
    resize_compress = args.resize_compress
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print_exp_details(args)
    print("Model: ", model)
    print("J =", J)
    print("a =", a)
    print("b =", b)
    print("QF_Y =",QF_Y)
    print("QF_C =",QF_C)

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
    
    dataset = HDQ_loader(root="~/data/ImageNet/2012", QF_Y=QF_Y, QF_C=QF_C, J=J, a=a, b=b, split="val", resize_compress=resize_compress)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=False, num_workers=8)
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
            l = l0 + l1 + l2
            print_file(l, args.output_txt)
        cnt += 1

    l0 = "#"* 30 + "\n"
    l1 = str(num_correct/num_tests) + " = " + str(num_correct) + " / "+ str(num_tests) + "\n"
    l2 = str(BPP.numpy()/num_tests) + "\n"
    l = l0 + l1 + l2
    print_file(l, args.output_txt)



if '__main__' == __name__:
    parser = argparse.ArgumentParser(description="HDQ")
    parser.add_argument('--Model', type=str, default="Alexnet", help='Subsampling b')
    parser.add_argument('--J', type=int, default=4, help='Subsampling J')
    parser.add_argument('--a', type=int, default=4, help='Subsampling a')
    parser.add_argument('--b', type=int, default=4, help='Subsampling b')
    parser.add_argument('--QF_Y', type=int, default=50, help='Quality factor of Y channel')
    parser.add_argument('--QF_C', type=int, default=50, help='Quality factor of Cb & Cr channel')
    parser.add_argument('-resize_compress', action='store_true', help='For Resize --> Compress set True')
    parser.add_argument('--output_txt', type=str, help='output txt file')
    parser.add_argument('--device', type=str, default="cuda:0", help='cpu or cuda:0')
    args = parser.parse_args()
    main(args)