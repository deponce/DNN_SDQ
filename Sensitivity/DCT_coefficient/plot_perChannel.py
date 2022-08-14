import numpy as np
import matplotlib.pyplot as plt
from utils import *
from tqdm import tqdm
from scipy.stats import bootstrap
import argparse
import os
import glob

main_dir = "/home/h2amer/AhmedH.Salamah/workspace_pc15/JPEG_lin/DNN_SDQ/SenMap_All/SenMap_Scale/"


def plot_data_bootstrap(old_channel, channel_sen_img, channel, model):
    bottom_lst = []
    top_lst = []
    mean_lst = []

    file_name = main_dir+model+"_"+channel+".txt"
    print(file_name)
    
    for i in tqdm(range(64)):
        bottom,top = list(bootstrap((channel_sen_img[i],), np.mean, confidence_level=0.95,n_resamples=100).confidence_interval)
        mean = np.mean((bottom,top))
        bottom_lst.append(bottom)
        top_lst.append(top)       
        mean_lst.append(mean)

    mean_lst = np.array(mean_lst)

    print(mean_lst/old_channel)

    print(channel)
    for x in mean_lst:
        print(x)

    np.savetxt(file_name, mean_lst, "%.20e")

    plt.figure(figsize=(10,8))
    plt.plot(old_channel, label="old")
    plt.xticks(np.arange(1,65,4))
    plt.legend()
    plt.title(channel+'L1 sensitivity, per image')
    plt.savefig(channel+model+".png", dpi=600)
    plt.figure().clear()

    plt.figure(figsize=(10,8))
    plt.plot(mean_lst, label="NEW")
    plt.xticks(np.arange(1,65,4))
    plt.legend()
    plt.title(channel+'L1 sensitivity, per image')
    plt.savefig(channel+"_new_"+model+".png", dpi=600)
    plt.figure().clear()

def plot_data_mean(old_channel, channel_sen_img, channel_compare, channel, model):

    mean_lst = []
    for i in tqdm(range(64)):
        mean = np.mean(channel_sen_img[i])
        mean_lst.append(mean)
    mean_lst = np.array(mean_lst)

    mean_lst_cmp = []
    for i in tqdm(range(64)):
        mean = np.mean(channel_compare[i])
        mean_lst_cmp.append(mean)

    mean_lst_cmp = np.array(mean_lst_cmp)

    print(mean_lst/mean_lst_cmp)

    print(channel)
    for x in mean_lst:
        print(x)

    plt.figure(figsize=(10,8))
    plt.plot(old_channel, label="old")
    plt.xticks(np.arange(1,65,4))
    plt.legend()
    plt.title(channel+'L1 sensitivity, per image')
    plt.savefig(channel+model+".png", dpi=600)
    plt.figure().clear()

    plt.figure(figsize=(10,8))
    plt.plot(mean_lst, label="NEW_1M")
    plt.plot(mean_lst_cmp, label="NEW_10K")
    plt.xticks(np.arange(1,65,4))
    plt.legend()
    plt.title(channel+'L1 sensitivity, per image')
    plt.savefig(channel+"_new_"+model+".png", dpi=600)
    plt.figure().clear()


def load(grad, model):
    Y_sen_list = np.load("./"+grad+"Y_sen_list_" + model + ".npy")
    Cb_sen_list = np.load("./"+grad+"Cb_sen_list_" + model + ".npy")
    Cr_sen_list = np.load("./"+grad+"Cr_sen_list_" + model + ".npy")
    zigzag = get_zigzag()
    lst_length = Y_sen_list.shape[0]
    Y_sen_img = np.zeros((64,lst_length))
    Cb_sen_img = np.zeros((64,lst_length))
    Cr_sen_img = np.zeros((64,lst_length))
    for i in range(8):
        for j in range(8):
            Y_sen_img[zigzag[i,j]] = Y_sen_list[:,i,j]
    del Y_sen_list
    for i in range(8):
        for j in range(8):
            Cb_sen_img[zigzag[i,j]] = Cb_sen_list[:,i,j]
    del Cb_sen_list
    for i in range(8):
        for j in range(8):
            Cr_sen_img[zigzag[i,j]] = Cr_sen_list[:,i,j]
    del Cr_sen_list
    return Y_sen_img, Cb_sen_img, Cr_sen_img


# read = "/home/h2amer/AhmedH.Salamah/workspace_pc15/JPEG_lin/DNN_SDQ/SenMap_All/SenMap_Scale_10K/"
read = "/home/h2amer/AhmedH.Salamah/workspace_pc15/JPEG_lin/DNN_SDQ/SenMap_All_nll/SenMap_Scale/"



sen_map = []
def main(model = 'Alexnet'):
    files = glob.glob(read+model+"*_Cb_KLT.txt")[0]
    old_sel_cb = np.loadtxt(files)

    files = glob.glob(read+model+"*_Cr_KLT.txt")[0]
    old_sel_cr = np.loadtxt(files)

    files = glob.glob(read+model+"*_Y_KLT.txt")[0]
    old_sel_y = np.loadtxt(files)



    Y_sen_img_K, Cb_sen_img_K, Cr_sen_img_K = load(grad="grad/", model=model)
    # Y_sen_img_M, Cb_sen_img_M, Cr_sen_img_M = load(grad="grad_1M/", model=model)

    # plot_data = plot_data_mean
    # plot_data(old_channel=old_sel_y, channel_sen_img= Y_sen_img_M, channel_compare = Y_sen_img_K, channel="Y_Channel", model=model)
    # plot_data(old_channel=old_sel_cb, channel_sen_img= Cb_sen_img_M, channel_compare = Cb_sen_img_K, channel="Cb_Channel", model=model)
    # plot_data(old_channel=old_sel_cr, channel_sen_img= Cr_sen_img_M, channel_compare = Cr_sen_img_K, channel="Cr_Channel", model=model)
    
    plot_data = plot_data_bootstrap

    plot_data(old_channel=old_sel_y, channel_sen_img= Y_sen_img_K, channel="Y_Channel", model=model)
    plot_data(old_channel=old_sel_cb, channel_sen_img= Cb_sen_img_K, channel="Cb_Channel", model=model)
    plot_data(old_channel=old_sel_cr, channel_sen_img= Cr_sen_img_K, channel="Cr_Channel", model=model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('-model',type=str, default='Alexnet', help='DNN model')
    args = parser.parse_args()
    main(**vars(args))
