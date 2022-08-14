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
            # print(tmp)
            top1 = float(tmp[0])
            bpp = float(tmp[1])
            return top1, bpp
        if "*" in line:
            flag = True


def get_data(filename):
    list_files = glob.glob(filename)
    print(filename)
    print("Number of files: ", len(list_files))
    d_list = []
    top1_list = []
    bbp_list = []
    for file in list_files:
        d = file.split("d_water_Y")[1].split("_")[0]
        try:
            top1, bpp = findline(file)
        except:
            print(file)
            continue
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
    method1 = minRate
    method1_name = "minRate"
    method1_th = 5.5

    method2 = minAcc
    method2_name = "minAcc"
    method2_th = 69.5
    # for model in ["VGG11", "Alexnet", "Resnet18", "Squeezenet"]:
    for model in ["Resnet18"]:
        plt.figure()
        # sens = "NoModel"
        # filename = "./Resize_Compress/HDQ_OptD_correct_YUV_Qmax/"+model+"/YUV/"+model+"_sens_"+sens+"*.txt"
        # HDQ_OptD_d, HDQ_OptD_top1, HDQ_OptD_bbp = get_data(filename)
        # HDQ_OptD_top1, HDQ_OptD_bbp = method1(HDQ_OptD_top1, HDQ_OptD_bbp, method1_th)   
        # HDQ_OptD_top1, HDQ_OptD_bbp = method2(HDQ_OptD_top1, HDQ_OptD_bbp, method2_th)   
        HDQ_OptD_top1 = [69.57799999999999, 69.724, 69.732]
        HDQ_OptD_bbp = [5.665731779593825, 7.056494745534659, 9.814586733690202]
        # plt.scatter(HDQ_OptD_bbp, HDQ_OptD_top1,label="HDQ_NoModel", marker=".", linestyle= '--')
        plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="HDQ_OptD", marker="o", markersize=3, linestyle= '-')

        # sens = "SenMap_Normalized"
        # filename = "./Resize_Compress/HDQ_OptD_correct_YUV_Qmax/"+model+"/YUV/"+model+"_sens_"+sens+"*.txt"
        # HDQ_OptD_d, HDQ_OptD_top1, HDQ_OptD_bbp = get_data(filename)
        # HDQ_OptD_top1, HDQ_OptD_bbp = method1(HDQ_OptD_top1, HDQ_OptD_bbp, method1_th)   
        # HDQ_OptD_top1, HDQ_OptD_bbp = method2(HDQ_OptD_top1, HDQ_OptD_bbp, method2_th)  
        HDQ_OptD_top1 = [69.632, 69.71000000000001, 69.76599999999999, 69.768, 69.75200000000001]
        HDQ_OptD_bbp = [5.8386435563707355, 6.1578031436942515, 8.478319431611746, 8.556955354803353, 9.198021735099108]
        # plt.scatter(HDQ_OptD_bbp, HDQ_OptD_top1,label="HDQ_Norm", marker=".", linestyle= '--')
        plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="HDQ_OptD_Sen", marker="o", markersize=3, linestyle= '-')


        # filename = "./Resize_Compress/HDQ/YUV444/"+model+"_PC52/"+model+"_QF"+"*.txt"
        # HDQ_OptD_top1, HDQ_OptD_bbp = get_data_HDQ(filename)
        # HDQ_OptD_top1, HDQ_OptD_bbp = method1(HDQ_OptD_top1, HDQ_OptD_bbp, method1_th)   
        # HDQ_OptD_top1, HDQ_OptD_bbp = method2(HDQ_OptD_top1, HDQ_OptD_bbp, method2_th)  
        HDQ_OptD_top1 = [69.556, 69.69999999999999, 69.708, 69.716]
        HDQ_OptD_bbp = [5.89708804907769, 6.837979052327722, 8.565019512687028, 10.104755549942702]
        # plt.scatter(HDQ_OptD_bbp, HDQ_OptD_top1,label="HDQ", marker=".", linestyle= '--')
        plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="HDQ", marker="o", markersize=3, linestyle= '-')

        plt.hlines(y=69.758, xmin=5.5, xmax=10.5, label="Top-1 Acc",alpha=0.5) 

        plt.title(model)
        plt.xlabel("rate(bpp)")
        plt.ylabel("accuracy(%)")
        plt.legend()
        plt.grid(linestyle='--', linewidth=0.5)
        plt.savefig("HDQ_OptD_"+model+ "_" + method1_name + "+"+ method2_name+".png",dpi=600)



if __name__ == "__main__":
    main()
      
      
       
      



   
