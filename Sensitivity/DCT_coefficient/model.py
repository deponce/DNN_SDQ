from torchvision import models
import torch


def get_model(name, device):
    if name =='Alexnet':
        pretrained_model = models.alexnet(pretrained=True)
    elif name == 'Resnet18':
        pretrained_model = models.resnet18(pretrained=True)
    elif name == 'Squeezenet':
        pretrained_model = models.squeezenet1_0(pretrained=True)
    elif name == 'VGG11':
        pretrained_model = models.vgg11(pretrained=True)
    elif name == 'mobilenet_v2':
        pretrained_model = models.mobilenet_v2(pretrained=True)
    elif name == 'ViT_B_16':
        pretrained_model = models.vision_transformer.vit_b_16(pretrained=True)
    elif name == 'Densenet':
        pretrained_model = models.densenet161(pretrained=True)
    elif name == 'Googlenet':
        pretrained_model = models.googlenet(pretrained=True)
    elif name == 'Shufflenetv2':
        pretrained_model = models.shufflenet_v2_x1_0(pretrained=True)
    elif name == 'Efficientnet':
        pretrained_model = models.efficientnet_b0(pretrained=True)
    elif name == 'Regnet':
        pretrained_model = models.regnet_x_16gf(pretrained=True)
    elif name == 'Mnasnet':
        pretrained_model = models.mnasnet1_0(pretrained=True)   
    elif name == 'mobilenet_v3':
        pretrained_model = models.mobilenet_v3_large(pretrained=True)
    return pretrained_model
    