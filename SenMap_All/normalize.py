import os
import re
import numpy as np
import matplotlib.pyplot as plt

# beta = 500000
# read = "./SenMap_Scale_10K/"
# read = "./SenMap_Scale_allSet/"
read = "./SenMap_Scale/"

# norm1 = "./SenMap_Scale_Normalized1/"
# norm2 = "./SenMap_Scale_Normalized2/"
# norm3 = "./SenMap_Scale_Normalized3/"
# norm4 = "./SenMap_Scale_Norm/"
# norm5 = "./SenMap_Normalized/"
norm5 = "./SenMap_Normalized/"
# norm5 = "./SenMap_Normalized_10K/"
files = os.listdir(read)

# # normalize_1: Linfeng's method
# for f in files:
#     sen_map = np.loadtxt(read+f)
#     if not re.findall(r"NoModel.*", f):
#         # print(sen_map)
#         Range = np.max(sen_map) - np.min(sen_map)
#         sen_map = sen_map / Range
#         mean_val = np.mean(sen_map)
#         sen_map = sen_map - mean_val + 1
#         # print(sen_map)
#         # print(sen_map.mean())
#         # print(sen_map.std())
#         if min(sen_map) > 0:
#             np.savetxt(norm1+f, sen_map, "%f")
#     else:
#         np.savetxt(norm1+f, sen_map, "%f")

# # normalize_2: mean = 1, std = 0.3
# for f in files:
#     sen_map = np.loadtxt(read+f)
#     if not re.findall(r"NoModel.*", f):
#         # print(sen_map)
#         sen_map = (sen_map-sen_map.mean())/(10/3*sen_map.std())+1
#         # print(sen_map)
#         # print(sen_map.mean())
#         # print(sen_map.std())
#         if min(sen_map) > 0:
#             np.savetxt(norm2+f, sen_map, "%f")
#     else:
#         np.savetxt(norm2+f, sen_map, "%f")

# # normalize_3: weighted softmax
# for f in files:
#     sen_map = np.loadtxt(read+f)
#     if not re.findall(r"NoModel.*", f):
#         # print(sen_map)
#         sen_map = np.exp(sen_map*beta)/sum(np.exp(sen_map*beta))
#         # print(sen_map)
#         # print(sen_map.mean())
#         # print(sen_map.std())
#         if min(sen_map) > 0:
#             np.savetxt(norm3+f, sen_map, "%f")
#     else:
#         np.savetxt(norm3+f, sen_map, "%f")

# # vanilla normalization
# for f in files:
#     sen_map = np.loadtxt(read+f)
#     if not re.findall(r"NoModel.*", f):
#         # print(sen_map)
#         sen_map = sen_map/max(sen_map)
#         # print(sen_map)
#         # print(sen_map.mean())
#         # print(sen_map.std())
#         if min(sen_map) > 0:
#             np.savetxt(norm4+f, sen_map, "%f")
#     else:
#         np.savetxt(norm4+f, sen_map, "%f")

# plt.plot(sen_map_org,alpha=0.5,label="Original")
# plt.figure()
# plt.plot(sen_map,alpha=0.5,label="Scaled Senstivity Maop")
# plt.legend()
# plt.show()

# same factor over 3 channels
# norm_res = 10*max(np.loadtxt(read+"Resnet18_Y.txt"))
# print("norm_res =", norm_res)
# norm_alex = 10*max(np.loadtxt(read+"Alexnet_Y.txt"))
# print("norm_alex =", norm_alex)
# norm_vgg = 10*max(np.loadtxt(read+"VGG11_Y.txt"))
# print("norm_vgg =", norm_vgg)
# norm_squeeze = 10*max(np.loadtxt(read+"Squeezenet_Y.txt"))
# print("norm_squeeze =", norm_squeeze)
# # norm_mobilenet_v2 = 10*max(np.loadtxt(read+"Mobilenet_v2_Y.txt"))
# # print("norm_mobilenet_v2 =", norm_mobilenet)
# norm_reg = 10*max(np.loadtxt(read+"Regnet_Y.txt"))
# print("norm_Regnet =", norm_reg)
# norm_vit = 10*max(np.loadtxt(read+"ViT_B_16_Y.txt"))
# print("norm_vit =", norm_vit)
# norm_Shufflenetv2 = 10*max(np.loadtxt(read+"Shufflenetv2_Y.txt"))
# print("norm_Shufflenetv2 =", norm_Shufflenetv2)
norm_Mnasnet = 10*max(np.loadtxt(read+"Mnasnet_Y.txt"))
print("norm_Mnasnet =", norm_Mnasnet)

for f in files:
    sen_map = np.loadtxt(read+f)
    # if re.findall(r"Resnet18.*", f):
    #     sen_map = sen_map/norm_res
    #     np.savetxt(norm5+f, sen_map, "%f")
    # if re.findall(r"NoModel.*", f):
    #     np.savetxt(norm5+f, sen_map, "%f")
    # if re.findall(r"Alexnet.*", f):
    #     sen_map = sen_map/norm_alex
    #     np.savetxt(norm5+f, sen_map, "%f")
    # if re.findall(r"VGG11.*", f):
    #     sen_map = sen_map/norm_vgg
    #     np.savetxt(norm5+f, sen_map, "%f")
    # if re.findall(r"Squeezenet.*", f):
    #     sen_map = sen_map/norm_squeeze
    #     np.savetxt(norm5+f, sen_map, "%f")
    # # if re.findall(r"Mobilenet_v2.*", f):
    # #     sen_map = sen_map/norm_mobilenet_v2
    # #     np.savetxt(norm5+f, sen_map, "%f")
    # if re.findall(r"Regnet.*", f):
    #     sen_map = sen_map/norm_reg
    #     np.savetxt(norm5+f, sen_map, "%f")
    # if re.findall(r"ViT_B_16.*", f):
    #     sen_map = sen_map/norm_vit
    #     np.savetxt(norm5+f, sen_map, "%f")
    # if re.findall(r"Shufflenetv2.*", f):
    #     sen_map = sen_map/norm_Shufflenetv2
    #     np.savetxt(norm5+f, sen_map, "%f")
    if re.findall(r"Mnasnet.*", f):
        sen_map = sen_map/norm_Mnasnet
        np.savetxt(norm5+f, sen_map, "%f")
# for f in files:
#     org_sen_map = np.loadtxt(read+f)
#     norm_sen_map = np.loadtxt(norm5+f)
#     print(f)
#     print(org_sen_map[0]/norm_sen_map[0])
