import os
import re
import numpy as np
import matplotlib.pyplot as plt

beta = 500000
read = "./SenMap_Scale/"
norm1 = "./SenMap_Scale_Normalized1/"
norm2 = "./SenMap_Scale_Normalized2/"
norm3 = "./SenMap_Scale_Normalized3/"
norm4 = "./SenMap_Scale_Norm/"
files = os.listdir(read)

# normalize_1: Linfeng's method
for f in files:
    sen_map = np.loadtxt(read+f)
    if not re.findall(r"NoModel.*", f):
        # print(sen_map)
        Range = np.max(sen_map) - np.min(sen_map)
        sen_map = sen_map / Range
        mean_val = np.mean(sen_map)
        sen_map = sen_map - mean_val + 1
        # print(sen_map)
        # print(sen_map.mean())
        # print(sen_map.std())
        if min(sen_map) > 0:
            np.savetxt(norm1+f, sen_map, "%f")
    else:
        np.savetxt(norm1+f, sen_map, "%f")


# normalize_2: mean = 1, std = 0.3
for f in files:
    sen_map = np.loadtxt(read+f)
    if not re.findall(r"NoModel.*", f):
        # print(sen_map)
        sen_map = (sen_map-sen_map.mean())/(10/3*sen_map.std())+1
        # print(sen_map)
        # print(sen_map.mean())
        # print(sen_map.std())
        if min(sen_map) > 0:
            np.savetxt(norm2+f, sen_map, "%f")
    else:
        np.savetxt(norm2+f, sen_map, "%f")


# normalize_3: weighted softmax
for f in files:
    sen_map = np.loadtxt(read+f)
    if not re.findall(r"NoModel.*", f):
        # print(sen_map)
        sen_map = np.exp(sen_map*beta)/sum(np.exp(sen_map*beta))
        # print(sen_map)
        # print(sen_map.mean())
        # print(sen_map.std())
        if min(sen_map) > 0:
            np.savetxt(norm3+f, sen_map, "%f")
    else:
        np.savetxt(norm3+f, sen_map, "%f")


# normalize_4: 1st norm
for f in files:
    sen_map = np.loadtxt(read+f)
    # print(f)
    sen_map_org = sen_map
    if not re.findall(r"NoModel.*", f):
        if min(sen_map) > 0:
            sen_map = sen_map/np.linalg.norm(sen_map, ord=1)
            sen_map_min = np.min(sen_map)
            while(sen_map_min < 1):
                sen_map = sen_map *10
                sen_map_min = np.min(sen_map)
            print(sen_map[0]/sen_map_org[0])
            np.savetxt(norm4+f, sen_map, "%f")
    else:
        np.savetxt(norm4+f, sen_map, "%f")

# plt.plot(sen_map_org,alpha=0.5,label="Original")
# plt.figure()
# plt.plot(sen_map,alpha=0.5,label="Scaled Senstivity Maop")
# plt.legend()
# plt.show()