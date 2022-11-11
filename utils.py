import torch 
from torchvision import models

def load_model(Model):
    if Model=="Alexnet":
        pretrained_model = models.alexnet(pretrained=True).eval()
    elif Model=="Resnet18":
        pretrained_model = models.resnet18(pretrained=True).eval()
    elif Model=="VGG11":  
        pretrained_model = models.vgg11(pretrained=True).eval()
    elif Model=="Squeezenet":
        pretrained_model = models.squeezenet1_0(pretrained=True).eval()
    elif Model=="NoModel":
        pretrained_model = models.vgg11(pretrained=True).eval()
    elif Model == 'ViT_B_16':
        pretrained_model = models.vision_transformer.vit_b_16(pretrained=True).eval()
    elif Model == 'Shufflenetv2':
        pretrained_model = models.shufflenet_v2_x1_0(pretrained=True).eval()
    elif Model == 'Regnet':
        pretrained_model = models.regnet_x_16gf(pretrained=True).eval()
    elif Model == 'Mnasnet':
        pretrained_model = models.mnasnet1_0(pretrained=True).eval()
    elif Model == 'mobilenet_v2':
        pretrained_model = models.mobilenet_v2(pretrained=True).eval()
    else: 
        print("Enter a model SOS")
        exit(0)
    return pretrained_model

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().sum(dtype=torch.float32)
        # correct_k = torch.reshape(correct[:k],(1, -1))
        # correct_k = torch.squeeze(correct_k)
        # correct_k = correct_k.float().sum(0)
        # res.append(correct_k.mul_(100.0 / batch_size))
        res.append(correct_k)
    return res

def compute_accuracy(outputs, targets, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        _, preds = outputs.topk(maxk, 1, True, True)
        preds = preds.t()
        corrects = preds.eq(targets[None])

        result_list = []
        for k in topk:
            # print(corrects[:k].flatten().sum(dtype=torch.float32), batch_size)
            correct_k = corrects[:k].flatten().sum(dtype=torch.float32)
            # result_list.append(correct_k * (100.0 / batch_size))
            result_list.append(correct_k)
        return result_list

def print_file(l1, output_txt):
    print(l1)
    f = open(output_txt, "+a")
    f.write(l1)
    f.close()

def print_exp_details(args):
    f = open(args.output_txt, "+a")
    f.write(args.device+ "\n")
    f.write("Model: "+ args.Model+ "\n")
    f.write("ColorSpace: "+ str(args.colorspace)+ "\n")
    f.write("J = "+ str(args.J)+ "\n")
    f.write("a = "+ str(args.a)+ "\n")
    f.write("b = "+ str(args.b)+ "\n")
    f.write("QF_Y = "+ str(args.QF_Y)+ "\n")
    f.write("QF_C = "+ str(args.QF_C)+ "\n")
    f.close()

def print_exp_details_SDQ(args):
    f = open(args.output_txt, "+a")
    f.write(args.device+ "\n")
    f.write("Model: "+ args.Model+ "\n")
    f.write("ColorSpace: "+ str(args.colorspace)+ "\n")
    f.write("J = "+ str(args.J)+ "\n")
    f.write("a = "+ str(args.a)+ "\n")
    f.write("b = "+ str(args.b)+ "\n")
    f.write("QF_Y = "+ str(args.QF_Y) + "\n")
    f.write("QF_C = "+ str(args.QF_C) + "\n")
    f.write("Beta_S = "+ str(args.Beta_S) + "\n")
    f.write("Beta_W = "+ str(args.Beta_W) + "\n")
    f.write("Beta_X = "+ str(args.Beta_X) + "\n")
    f.write("Lambda = "+ str(args.L) + "\n")
    f.close()

# def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.FloatTensor]:
#     list_topk_accs = []  # idx is topk1, topk2, ... etc
    
#     for k in topk:
#         # get tensor of which topk answer was right
#         ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
#         # flatten it to help compute if we got it correct for each example in batch
#         flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
#         # get if we got it right for any of our top k prediction for each example in batch
#         tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
#         # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
#         topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
#         list_topk_accs.append(topk_acc)
#     return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]

