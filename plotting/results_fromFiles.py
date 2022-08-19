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
    method2_th = 68
    # for model in ["VGG11", "Alexnet", "Resnet18", "Squeezenet"]:
    for model in ["VGG11"]:
        plt.figure()
        # sens = "NoModel"
        # filename = "./Resize_Compress/HDQ_OptD_correct_YUV_Qmax/"+model+"/YUV/"+model+"_sens_"+sens+"*.txt"
        # HDQ_OptD_d, HDQ_OptD_top1, HDQ_OptD_bbp = get_data(filename)
        # HDQ_OptD_top1, HDQ_OptD_bbp = method1(HDQ_OptD_top1, HDQ_OptD_bbp, method1_th)   
        # HDQ_OptD_top1, HDQ_OptD_bbp = method2(HDQ_OptD_top1, HDQ_OptD_bbp, method2_th)   
        # HDQ_OptD_top1 = [69.57799999999999, 69.724, 69.732]
        # HDQ_OptD_bbp = [5.665731779593825, 7.056494745534659, 9.814586733690202]
        # plt.scatter(HDQ_OptD_bbp, HDQ_OptD_top1,label="HDQ_NoModel", marker=".", linestyle= '--')
        # plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="YUV_HDQ_OptD", marker="o", markersize=3, linestyle= '-')

        # sens = "SenMap_Normalized"
        # filename = "./Resize_Compress/HDQ_OptD_correct_YUV_Qmax/"+model+"/YUV/"+model+"_sens_"+sens+"*.txt"
        # HDQ_OptD_d, HDQ_OptD_top1, HDQ_OptD_bbp = get_data(filename)
        # HDQ_OptD_top1, HDQ_OptD_bbp = method1(HDQ_OptD_top1, HDQ_OptD_bbp, method1_th)   
        # HDQ_OptD_top1, HDQ_OptD_bbp = method2(HDQ_OptD_top1, HDQ_OptD_bbp, method2_th)  
        # plt.scatter(HDQ_OptD_bbp, HDQ_OptD_top1,label="HDQ_Norm", marker=".", linestyle= '--')
        # plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="HDQ_OptD_Sen", marker="o", markersize=3, linestyle= '-')

        # filename = "./Resize_Compress/HDQ/YUV444/"+model+"_PC52/"+model+"_QF"+"*.txt"
        # HDQ_OptD_top1, HDQ_OptD_bbp = get_data_HDQ(filename)
        # HDQ_OptD_top1, HDQ_OptD_bbp = method1(HDQ_OptD_top1, HDQ_OptD_bbp, method1_th)   
        # HDQ_OptD_top1, HDQ_OptD_bbp = method2(HDQ_OptD_top1, HDQ_OptD_bbp, method2_th)  
        # plt.scatter(HDQ_OptD_bbp, HDQ_OptD_top1,label="YUV444", marker=".", linestyle= '--')

        HDQ_OptD_top1 = [69.75, 69.71000000000001, 69.684, 69.66]
        HDQ_OptD_bbp = [10.294984782875776, 7.334578795809299, 6.475910036938339, 5.809912195831091]
        plt.scatter(HDQ_OptD_bbp, HDQ_OptD_top1,label="SWX444_OptD")

        HDQ_OptD_top1 = [69.362, 69.202, 69.19800000000001, 69.27]
        HDQ_OptD_bbp = [6.556066576657742, 4.082690363262147, 4.468203978889138,4.959945770005732]
        plt.scatter(HDQ_OptD_bbp, HDQ_OptD_top1,label="SWX420_OptD")

        # filename = "./Resize_Compress/HDQ/YUV420/"+model+"_PC52/"+model+"_QF"+"*.txt"
        # HDQ_OptD_top1, HDQ_OptD_bbp = get_data_HDQ(filename)
        # # HDQ_OptD_top1, HDQ_OptD_bbp = method1(HDQ_OptD_top1, HDQ_OptD_bbp, method1_th)   
        # HDQ_OptD_top1, HDQ_OptD_bbp = method2(HDQ_OptD_top1, HDQ_OptD_bbp, method2_th)  
        # plt.scatter(HDQ_OptD_bbp, HDQ_OptD_top1,label="YUV420", marker=".", linestyle= '--')
        # # plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="YUV420_HDQ")

        # plt.hlines(y=69.02, xmin=min(HDQ_OptD_bbp), xmax=max(HDQ_OptD_bbp), label="Top-1 Acc",alpha=0.5) 

        plt.title(model)
        plt.xlabel("rate(bpp)")
        plt.ylabel("accuracy(%)")
        plt.legend()
        plt.grid(linestyle='--', linewidth=0.5)
        # plt.savefig("HDQ_OptD_"+model+ "_" + method1_name + "+"+ method2_name+".png",dpi=600)
        plt.savefig("SWX_compare_subsampling.png",dpi=600)



if __name__ == "__main__":
    main()
      
      
       
      



   
