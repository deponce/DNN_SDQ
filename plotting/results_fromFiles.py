import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.neighbors import NearestNeighbors


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

def interval_best(acc, bpp, n):
    new_acc=np.zeros(n)
    new_bpp=np.zeros(n)
    max_bpp=np.max(bpp)
    min_bpp=np.min(bpp)
    delta=(max_bpp-min_bpp)/n
    for i in range(len(bpp)):
        for j in range(n):
            if (bpp[i]>=(min_bpp+j*delta)) and (bpp[i]<(min_bpp+(j+1)*delta)):
                if acc[i]>new_acc[j]:
                    new_acc[j]=acc[i]
                    new_bpp[j]=bpp[i]
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

# def find_nearestbpp():



# def convex_plot(HDQ_bbp, HDQ_top1, HDQ_OptD_G_bbp, HDQ_OptD_G_top1, HDQ_OptD_top1, HDQ_OptD_bbp):

#     # arg_indx = sorted(range(len(HDQ_bbp)), key=lambda k: HDQ_bbp[k])
#     breakpoint()
    

#     return HDQ_bbp, HDQ_top1, HDQ_OptD_G_bbp, HDQ_OptD_G_top1, HDQ_OptD_top1, HDQ_OptD_bbp

def main():
    # # VGG11_sens_NoModel_d_water_Y.10_Q_max_Y255
    # # VGG11_sens_SenMap_Normalized_d_water_Y3.40_Q_max_Y255
    # method1 = minRate
    # method1_name = "minRate"
    # method1_th = 2         # VGG11, Resnet18, Mobilenet_v2, Regnet
    # # method1_th = 5        # Alexnet


    # method2 = minAcc
    # method2_name = "minAcc"
    # method2_th = 0

    # method3 = maxRate
    # method3_name = "maxRate"
    # method3_th = 5
    # # for model in ["VGG11", "Alexnet", "Resnet18", "Mobilenet_v2"]:
    # for model in ["mobilenet_v2"]:
    #     plt.figure()
    #     # sens = "NoModel"
    #     # filename = "./Resize_Compress/HDQ_OptD_new_senmap_low_rate/"+model+"/YUV/"+model+"_sens_"+sens+"*.txt"
    #     # HDQ_OptD_d, HDQ_OptD_top1, HDQ_OptD_bbp = get_data(filename)
    #     # HDQ_OptD_top1, HDQ_OptD_bbp = method1(HDQ_OptD_top1, HDQ_OptD_bbp, method1_th)   
    #     # HDQ_OptD_top1, HDQ_OptD_bbp = method2(HDQ_OptD_top1, HDQ_OptD_bbp, method2_th)   
    #     # # HDQ_OptD_top1, HDQ_OptD_bbp = method3(HDQ_OptD_top1, HDQ_OptD_bbp, method3_th)  
    #     # plt.scatter(HDQ_OptD_bbp, HDQ_OptD_top1,label="HDQ+OptD", marker=".", linestyle= '--', alpha=0.5)
    #     # # plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="YUV_HDQ_OptD", marker="o", markersize=3, linestyle= '-')

    #     sens = "SenMap_Normalized"
    #     filename = "./Resize_Compress/HDQ_OptD_new_senmap_low_rate/"+model+"/YUV/"+model+"_sens_"+sens+"*.txt"
    #     HDQ_OptD_G_d, HDQ_OptD_G_top1, HDQ_OptD_G_bbp = get_data(filename)
    #     HDQ_OptD_G_top1, HDQ_OptD_G_bbp = method1(HDQ_OptD_G_top1, HDQ_OptD_G_bbp, method1_th)   
    #     HDQ_OptD_G_top1, HDQ_OptD_G_bbp = method3(HDQ_OptD_G_top1, HDQ_OptD_G_bbp, method3_th)  
    #     HDQ_OptD_G_top1, HDQ_OptD_G_bbp = interval_best(HDQ_OptD_G_top1, HDQ_OptD_G_bbp, 17) 
    #     # HDQ_OptD_G_top1, HDQ_OptD_G_bbp = method3(HDQ_OptD_G_top1, HDQ_OptD_G_bbp, method3_th)  
    #     plt.scatter(HDQ_OptD_G_bbp, HDQ_OptD_G_top1,label="HDQ+OptD(G)", marker=".", linestyle= '--', alpha=0.5)
    #     print(HDQ_OptD_G_top1)
    #     print(HDQ_OptD_G_bbp)
    #     # plt.plot(HDQ_OptD_G_bbp, HDQ_OptD_G_top1,label="HDQ_OptD_Sen", marker="o", markersize=3, linestyle= '-')

    #     filename = "./Resize_Compress/HDQ/YUV444/"+model+"_PC52/"+model+"_QF"+"*.txt"
    #     HDQ_top1, HDQ_bbp = get_data_HDQ(filename)
    #     HDQ_top1, HDQ_bbp = method1(HDQ_top1, HDQ_bbp, method1_th)   
    #     HDQ_top1, HDQ_bbp = method3(HDQ_top1, HDQ_bbp, method3_th)  
    #     plt.scatter(HDQ_bbp, HDQ_top1,label="HDQ", marker=".", linestyle= '--',  alpha=0.5)


    #     # HDQ_bbp, HDQ_top1, HDQ_OptD_G_bbp, HDQ_OptD_G_top1, HDQ_OptD_top1, HDQ_OptD_bbp \
    #     #         = convex_plot(HDQ_bbp, HDQ_top1, HDQ_OptD_G_bbp, HDQ_OptD_G_top1, HDQ_OptD_top1, HDQ_OptD_bbp)

    #     # plt.hlines(y=69.752, xmin=min(HDQ_bbp), xmax=max(HDQ_bbp), label="Top-1 Acc",alpha=0.5) 

    #     plt.title(model)
    #     plt.xlabel("rate(bpp)")
    #     plt.ylabel("accuracy(%)")
    #     plt.legend()
    #     # plt.grid(linestyle='--', linewidth=0.5)
    #     # plt.savefig("HDQ_OptD_"+model+ "_" + method1_name + "+"+ method2_name+".png",dpi=600)
    #     plt.savefig(model+"_HDQ_OptD_compare_low.png",dpi=1024)
    #     plt.savefig(model+"_HDQ_OptD_compare_low.pdf",dpi=1024)

    ##########################
    # # Resnet18

    # plt.figure()
    # HDQ_OptD_top1 = [69.658, 69.696, 69.774]
    # HDQ_OptD_bbp = [4.833476868264079, 6.11972353104502, 8.45331796366021]
    # plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="OptD(G)", marker="o", markersize=3, linestyle= '-')
    # HDQ_OptD_top1 = [69.598, 69.704, 69.734]
    # HDQ_OptD_bbp = [5.6678434691467885, 7.066747210933417, 9.814320875927509]
    # plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="OptD", marker="o", markersize=3, linestyle= '-')
    # filename = "./Resize_Compress/HDQ/YUV444/Resnet18_PC52/Resnet18_QF"+"*.txt"
    # HDQ_OptD_top1, HDQ_OptD_bbp = get_data_HDQ(filename)
    # HDQ_OptD_top1, HDQ_OptD_bbp = minRate(HDQ_OptD_top1, HDQ_OptD_bbp, 5.5)   
    # tmp1 = [HDQ_OptD_top1[0], HDQ_OptD_top1[2], HDQ_OptD_top1[1]]
    # tmp2 = [HDQ_OptD_bbp[0], HDQ_OptD_bbp[2], HDQ_OptD_bbp[1]]
    # plt.plot(tmp2, tmp1,label="JPEG", marker="o", markersize=3, linestyle= '-')
    # plt.hlines(y=69.758, xmin=4.833476868264079, xmax=max(HDQ_OptD_bbp), label="Raw Images",alpha=0.5) 
    # plt.title("ResNet18")
    # plt.xlabel("Rate (bpp)")
    # plt.ylabel("Accuracy (%)")
    # plt.legend(loc='lower right')
    # plt.grid(linestyle='--', linewidth=0.5)
    # plt.savefig("./paper_figures/Resnet18_HDQ_OptD_compare_3curves.pdf",dpi=1000)
    # plt.savefig("./paper_figures/Resnet18_HDQ_OptD_compare_3curves.png",dpi=1000)

    # ##########################
    # # Alexnet

    # plt.figure()
    # HDQ_OptD_top1 = [56.468, 56.538, 56.54]
    # HDQ_OptD_bbp = [4.885181616943776, 5.816937678545416, 8.673999072886556]
    # plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="OptD(G)", marker="o", markersize=3, linestyle= '-')
    # HDQ_OptD_top1 = [56.458, 56.53, 56.512]
    # HDQ_OptD_bbp = [5.7362531790751214, 7.150904317600727, 10.097702182869314]
    # plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="OptD", marker="o", markersize=3, linestyle= '-')
    # filename = "./Resize_Compress/HDQ/YUV444/Alexnet_PC52/Alexnet_QF"+"*.txt"
    # HDQ_OptD_top1, HDQ_OptD_bbp = get_data_HDQ(filename)
    # HDQ_OptD_top1, HDQ_OptD_bbp = minRate(HDQ_OptD_top1, HDQ_OptD_bbp, 5)   
    # tmp1 = [HDQ_OptD_top1[2], HDQ_OptD_top1[4], HDQ_OptD_top1[1]]
    # tmp2 = [HDQ_OptD_bbp[2], HDQ_OptD_bbp[4], HDQ_OptD_bbp[1]]
    # plt.plot(tmp2, tmp1,label="JPEG", marker="o", markersize=3, linestyle= '-')
    # plt.hlines(y=56.518, xmin= 4.885181616943776, xmax=max(HDQ_OptD_bbp), label="Raw Images",alpha=0.5) 
    # plt.title("AlexNet")
    # plt.xlabel("Rate (bpp)")
    # plt.ylabel("Accuracy (%)")
    # plt.legend(loc='lower right')
    # plt.grid(linestyle='--', linewidth=0.5)
    # plt.savefig("./paper_figures/Alexnet_HDQ_OptD_compare_3curves.pdf",dpi=1000)
    # plt.savefig("./paper_figures/Alexnet_HDQ_OptD_compare_3curves.png",dpi=1000)

    # ##########################
    # # VGG11

    # plt.figure()
    # HDQ_OptD_top1 = [69.04, 68.938, 68.872]
    # HDQ_OptD_bbp = [8.650134950927942, 6.427399690419137, 4.95179700579673]
    # plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="OptD(G)", marker="o", markersize=3, linestyle= '-')
    # HDQ_OptD_top1 = [69.002, 68.944, 68.892]
    # HDQ_OptD_bbp = [10.047948008631915, 7.064668663414568, 5.720029195137024]
    # plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="OptD", marker="o", markersize=3, linestyle= '-')
    # filename = "./Resize_Compress/HDQ/YUV444/VGG11_PC52/VGG11_QF"+"*.txt"
    # HDQ_OptD_top1, HDQ_OptD_bbp = get_data_HDQ(filename)
    # HDQ_OptD_top1, HDQ_OptD_bbp = minRate(HDQ_OptD_top1, HDQ_OptD_bbp, 5.5)   
    # tmp1 = [HDQ_OptD_top1[2], HDQ_OptD_top1[1], HDQ_OptD_top1[3]]
    # tmp2 = [HDQ_OptD_bbp[2], HDQ_OptD_bbp[1], HDQ_OptD_bbp[3]]
    # plt.plot(tmp2, tmp1,label="JPEG", marker="o", markersize=3, linestyle= '-')
    # plt.hlines(y=69.02, xmin= 4.95179700579673, xmax=max(HDQ_OptD_bbp), label="Raw Images",alpha=0.5) 
    # plt.title("VGG11")
    # plt.xlabel("Rate (bpp)")
    # plt.ylabel("Accuracy (%)")
    # plt.legend(loc='lower right')
    # plt.grid(linestyle='--', linewidth=0.5)
    # plt.savefig("./paper_figures/VGG11_HDQ_OptD_compare_3curves.pdf",dpi=1000)
    # plt.savefig("./paper_figures/VGG11_HDQ_OptD_compare_3curves.png",dpi=1000)

    # ##########################
    # # Regnet

    # plt.figure()
    # HDQ_OptD_top1 = [80.084, 79.998, 79.9]
    # HDQ_OptD_bbp = [9.578220539101808, 6.927074842850864, 5.22017788459152]
    # plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="OptD(G)", marker="o", markersize=3, linestyle= '-')
    # HDQ_OptD_top1 = [80.064, 80.0, 79.89]
    # HDQ_OptD_bbp = [9.90297594386667, 7.188149643199891, 5.726476511685252]
    # plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="OptD", marker="o", markersize=3, linestyle= '-')
    # filename = "./Resize_Compress/HDQ/YUV444/Regnet_PC52/Regnet_QF"+"*.txt"
    # HDQ_OptD_top1, HDQ_OptD_bbp = get_data_HDQ(filename)
    # HDQ_OptD_top1, HDQ_OptD_bbp = minRate(HDQ_OptD_top1, HDQ_OptD_bbp, 5.5)   
    # tmp1 = [HDQ_OptD_top1[0], HDQ_OptD_top1[1], HDQ_OptD_top1[3]]
    # tmp2 = [HDQ_OptD_bbp[0], HDQ_OptD_bbp[1], HDQ_OptD_bbp[3]]
    # plt.plot(tmp2, tmp1,label="JPEG", marker="o", markersize=3, linestyle= '-')
    # plt.hlines(y=80.058, xmin= 5.22017788459152, xmax=max(HDQ_OptD_bbp), label="Raw Images",alpha=0.5) 
    # plt.title("RegNet")
    # plt.xlabel("Rate (bpp)")
    # plt.ylabel("Accuracy (%)")
    # plt.legend(loc='lower right')
    # plt.grid(linestyle='--', linewidth=0.5)
    # plt.savefig("./paper_figures/Regnet_HDQ_OptD_compare_3curves.pdf",dpi=1000)
    # plt.savefig("./paper_figures/Regnet_HDQ_OptD_compare_3curves.png",dpi=1000)

    # ##########################
    # # Mnasnet

    # plt.figure()
    # HDQ_OptD_top1 = [73.49, 73.394, 73.386]
    # HDQ_OptD_bbp = [7.977118186638206, 5.34740031424731, 4.591961884927452]
    # plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="OptD(G)", marker="o", markersize=3, linestyle= '-')
    # HDQ_OptD_top1 = [73.448, 73.398, 73.388]
    # HDQ_OptD_bbp = [9.792116434750259, 7.09544279790923, 5.780553564886451]
    # plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="OptD", marker="o", markersize=3, linestyle= '-')
    # filename = "./Resize_Compress/HDQ/YUV444/Mnasnet_PC52/Mnasnet_QF"+"*.txt"
    # HDQ_OptD_top1, HDQ_OptD_bbp = get_data_HDQ(filename)
    # HDQ_OptD_top1, HDQ_OptD_bbp = minRate(HDQ_OptD_top1, HDQ_OptD_bbp, 5.5)   
    # tmp1 = [HDQ_OptD_top1[2], HDQ_OptD_top1[1], HDQ_OptD_top1[3]]
    # tmp2 = [HDQ_OptD_bbp[2], HDQ_OptD_bbp[1], HDQ_OptD_bbp[3]]
    # plt.plot(tmp2, tmp1,label="JPEG", marker="o", markersize=3, linestyle= '-')
    # plt.hlines(y=73.468, xmin= 4.591961884927452, xmax=max(HDQ_OptD_bbp), label="Raw Images",alpha=0.5) 
    # plt.title("MNASNet")
    # plt.xlabel("Rate (bpp)")
    # plt.ylabel("Accuracy (%)")
    # plt.legend(loc='lower right')
    # plt.grid(linestyle='--', linewidth=0.5)
    # plt.savefig("./paper_figures/Mnasnet_HDQ_OptD_compare_3curves.pdf",dpi=1000)
    # plt.savefig("./paper_figures/Mnasnet_HDQ_OptD_compare_3curves.png",dpi=1000)

    ##########################
    # Mobilenet_v2

    # plt.figure()
    # HDQ_OptD_top1 = [71.766, 71.884, 71.958]
    # HDQ_OptD_bbp = [4.798400177247227, 6.4191030776956675, 8.278568066779076]
    # plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="OptD(G)", marker="o", markersize=3, linestyle= '-')
    # HDQ_OptD_top1 = [71.754, 71.858, 71.884]
    # HDQ_OptD_bbp = [5.668460421251357, 7.150280818607211, 9.998163959823549]
    # plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="OptD", marker="o", markersize=3, linestyle= '-')
    # filename = "./Resize_Compress/HDQ/YUV444/Mobilenet_v2_PC15/Mobilenet_v2"+"*.txt"
    # HDQ_OptD_top1, HDQ_OptD_bbp = get_data_HDQ(filename) 
    # HDQ_OptD_top1, HDQ_OptD_bbp = minRate(HDQ_OptD_top1, HDQ_OptD_bbp, 5.5)
    # tmp1 = [HDQ_OptD_top1[0], HDQ_OptD_top1[3], HDQ_OptD_top1[1]]
    # tmp2 = [HDQ_OptD_bbp[0], HDQ_OptD_bbp[3], HDQ_OptD_bbp[1]]
    # plt.plot(tmp2, tmp1,label="JPEG", marker="o", markersize=3, linestyle= '-')
    # plt.hlines(y=71.878, xmin= 4.798400177247227, xmax=max(HDQ_OptD_bbp), label="Raw Images",alpha=0.5) 
    # plt.title("MobileNet_V2")
    # plt.xlabel("Rate (bpp)")
    # plt.ylabel("Accuracy (%)")
    # plt.legend(loc='lower right')
    # plt.grid(linestyle='--', linewidth=0.5)
    # plt.savefig("./paper_figures/Mobilenet_v2_HDQ_OptD_compare_3curves.pdf",dpi=1000)
    # plt.savefig("./paper_figures/Mobilenet_v2_HDQ_OptD_compare_3curves.png",dpi=1000)

    ##########################
    # # Mobilenet_v2 low rate

    # plt.figure()
    # HDQ_OptD_top1 = [70.226, 70.448, 70.714, 70.934, 71.124, 71.53, 71.586, 71.634, 71.682]
    # HDQ_OptD_bbp = [2.16585567, 2.340346, 2.50358779, 2.68419441, 3.00334573, 3.5541788, 3.87151267, 4.18484616, 4.63517147]
    # plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="OptD(G)", marker="o", markersize=3, linestyle= '-')
    # HDQ_OptD_top1 = [69.94, 70.322, 70.54, 70.838, 71.0, 71.3, 71.384, 71.442, 71.524]
    # HDQ_OptD_bbp = [2.164448947573006, 2.394443181256205, 2.5929309809204937, 2.8402573440366985, 3.1389417552007735, 3.495285343518406, 3.780690235467702, 4.140803396625667, 4.547156051995307]
    # plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="JPEG", marker="o", markersize=3, linestyle= '-')
    # plt.title("MobileNet_V2, low rate")
    # plt.xlabel("Rate (bpp)")
    # plt.ylabel("Accuracy (%)")
    # plt.legend(loc='lower right')
    # plt.grid(linestyle='--', linewidth=0.5)
    # plt.savefig("./paper_figures/Mobilenet_v2_lowrate.pdf",dpi=1000)
    # plt.savefig("./paper_figures/Mobilenet_v2_lowrate.png",dpi=1000)

    ##########################
    # # Mobilenet_v2 same SWE

    plt.figure()
    # HDQ_OptD_top1 = [69.702,69.726,69.832,70.074,70.246,70.374,70.456,70.568,70.686,70.736,71.026,71.102,71.188,71.356,71.446,71.518,71.624,71.66,71.768,71.822]
    # HDQ_OptD_bbp = [1.85,1.91,1.97,2.03,2.10,2.19,2.27,2.37,2.49,2.59,2.75,2.92,3.09,3.33,3.63,3.99,4.56,5.29,6.46,8.58]
    # plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="OptD(G)", marker="o", markersize=3, linestyle= '-')
    # HDQ_OptD_top1 = [69.776,69.94,70.014,70.162,70.322,70.426,70.54,70.672,70.838,70.862,71.0,71.126,71.3,71.384,71.442,71.524,71.644,71.718,71.84,71.872]
    # HDQ_OptD_bbp = [2.09,2.16,2.23,2.32,2.39,2.49,2.59,2.69,2.84,2.95,3.14,3.32,3.50,3.78,4.14,4.55,5.19,5.90,6.84,8.57]
    # plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="JPEG", marker="o", markersize=3, linestyle= '-')
    HDQ_OptD_top1 = [70.074,70.246,70.374,70.456,70.568,70.686,70.736,71.026,71.102,71.188,71.356,71.446,71.518,71.624]
    HDQ_OptD_bbp = [2.03,2.10,2.19,2.27,2.37,2.49,2.59,2.75,2.92,3.09,3.33,3.63,3.99,4.56]
    plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="OptD(G)", marker="o", markersize=3, linestyle= '-')
    HDQ_OptD_top1 = [70.162,70.322,70.426,70.54,70.672,70.838,70.862,71.0,71.126,71.3,71.384,71.442,71.524,71.644]
    HDQ_OptD_bbp = [2.32,2.39,2.49,2.59,2.69,2.84,2.95,3.14,3.32,3.50,3.78,4.14,4.55,5.19]
    plt.plot(HDQ_OptD_bbp, HDQ_OptD_top1,label="JPEG", marker="o", markersize=3, linestyle= '-')
    plt.title("MobileNet_V2, same SWE")
    plt.xlabel("Rate (bpp)")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc='best')
    plt.grid(linestyle='--', linewidth=0.5)
    plt.savefig("./paper_figures/Mobilenet_v2_same_SWE.pdf",dpi=1000)
    plt.savefig("./paper_figures/Mobilenet_v2_same_SWE.png",dpi=1000)
if __name__ == "__main__":
    main()
