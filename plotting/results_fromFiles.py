import numpy as np
import matplotlib.pyplot as plt
import glob



def minRate(acc, bpp, th):
    new_acc = []
    new_bpp = []
    for idx, b in enumerate(bpp):
        if b > th:
            new_acc.append(acc[idx])
            new_bpp.append(bpp[idx])
    return new_acc, new_bpp

def maxRate(acc, bpp, th):
    new_acc = []
    new_bpp = []
    for idx, b in enumerate(bpp):
        if b < th:
            new_acc.append(acc[idx])
            new_bpp.append(bpp[idx])
    return new_acc, new_bpp

def minAcc(acc, bpp, th):
    new_acc = []
    new_bpp = []
    for idx, Acc in enumerate(acc):
        if Acc > th:
            new_acc.append(acc[idx])
            new_bpp.append(bpp[idx])
    return new_acc, new_bpp

def maxAcc(acc, bpp, th):
    new_acc = []
    new_bpp = []
    for idx, Acc in enumerate(acc):
        if Acc < th:
            new_acc.append(acc[idx])
            new_bpp.append(bpp[idx])
    return new_acc, new_bpp

def concave(vgg_jpeg_acc, vgg_jpeg_bpp):
    new_acc = []
    new_bpp = []
    for idx, b in enumerate(bpp):
        if b > th:
            new_acc.append(acc[idx])
            new_bpp.append(bpp[idx])
    return new_acc, new_bpp


def findline(filename):
    f = open(filename,'r')
    line_index = 0
    flag = False
    for index , line in enumerate(f.readlines()):
        if flag:
            tmp = line.split("\t")
            top1 = float(tmp[0])
            bpp = float(tmp[1])
            return top1, bpp
        if "*" in line:
            flag = True


def get_data(filename):
    list_files = glob.glob(filename)
    print("Number of files: ", len(list_files))
    d_list = []
    top1_list = []
    bbp_list = []
    for file in list_files: 
        d = file.split("d_water_Y")[1].split("_")[0]
        top1, bpp = findline(file)
        d_list.append(d)
        top1_list.append(top1)
        bbp_list.append(bpp)
    return d_list, top1_list, bbp_list


def get_data_HDQ(filename):
    list_files = glob.glob(filename)
    print("Number of files: ", len(list_files))
    top1_list = []
    bbp_list = []
    for file in list_files: 
        top1, bpp = findline(file)
        top1_list.append(top1)
        bbp_list.append(bpp)
    return  top1_list, bbp_list


def main():
    # VGG11_sens_NoModel_d_water_Y.10_Q_max_Y255
    # VGG11_sens_SenMap_Normalized_d_water_Y3.40_Q_max_Y255
    method = minRate
    method_name = "minRate"
    for model in ["VGG11", "Alexnet", "Resnet18", "Squeezenet"]:
        plt.figure()
        sens = "NoModel"
        filename = "./Resize_Compress/HDQ_OptD/"+model+"/YUV/"+model+"_sens_"+sens+"*.txt"
        HDQ_OptD_d, HDQ_OptD_top1, HDQ_OptD_bbp = get_data(filename)
        HDQ_OptD_top1, HDQ_OptD_bbp = method(HDQ_OptD_top1, HDQ_OptD_bbp, 2)   
        plt.scatter(HDQ_OptD_bbp, HDQ_OptD_top1,label="HDQ_NoModel", marker=".", linestyle= '--')

        sens = "SenMap_Normalized"
        filename = "./Resize_Compress/HDQ_OptD/"+model+"/YUV/"+model+"_sens_"+sens+"*.txt"
        HDQ_OptD_d, HDQ_OptD_top1, HDQ_OptD_bbp = get_data(filename)
        HDQ_OptD_top1, HDQ_OptD_bbp = method(HDQ_OptD_top1, HDQ_OptD_bbp, 2) 
        plt.scatter(HDQ_OptD_bbp, HDQ_OptD_top1,label="HDQ_Norm", marker=".", linestyle= '--')

        filename = "./Resize_Compress/HDQ/YUV444/"+model+"_PC52/"+model+"_QF"+"*.txt"
        HDQ_OptD_top1, HDQ_OptD_bbp = get_data_HDQ(filename)
        HDQ_OptD_top1, HDQ_OptD_bbp = method(HDQ_OptD_top1, HDQ_OptD_bbp, 2) 
        plt.scatter(HDQ_OptD_bbp, HDQ_OptD_top1,label="HDQ", marker=".", linestyle= '--')

        # plt.hlines(y=69.02, xmin=0, xmax=max(HDQ_OptD_bbp), label="Top-1 Acc",alpha=0.5) 

        plt.title("VGG11")
        plt.xlabel("rate(bpp)")
        plt.ylabel("accuracy(%)")
        plt.legend()
        plt.grid(linestyle='--', linewidth=0.5)
        plt.savefig("HDQ_OptD_"+model+ "_" + method_name+".png",dpi=600)

if __name__ == "__main__":
    main()
