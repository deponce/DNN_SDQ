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
        Model = "VGG11"
    return pretrained_model


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