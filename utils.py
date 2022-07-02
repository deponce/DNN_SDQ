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