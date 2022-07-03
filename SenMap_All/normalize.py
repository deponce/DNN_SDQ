import os
import re
import numpy as np

beta = 500000
read = "./SenMap_Scale/"
norm1 = "./SenMap_Scale_Normalized1/"
norm2 = "./SenMap_Scale_Normalized2/"
norm3 = "./SenMap_Scale_Normalized3/"
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