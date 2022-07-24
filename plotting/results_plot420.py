import numpy as np
import matplotlib.pyplot as plt

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




vgg_jpeg_acc = [
29.854,
44.986,
51.1,
54.76,
57.466,
58.94,
60.054,
61.042,
61.612,
62.558,
63.298,
64.126,
64.674,
65.548,
66.278,
67.048,
67.9,
68.446,
68.814,
]

vgg_jpeg_bpp = [

0.3238483721,
0.4385142092,
0.5329366709,
0.6187172542,
0.7091290196,
0.7839219348,
0.8476006722,
0.9179742953,
0.9644029962,
1.046742993,
1.121270623,
1.214976776,
1.32423156,
1.45000145,
1.670050381,
1.942169868,
2.389122378,
3.288398905,
6.713584456,

]

vgg_HDQ_SWX_acc = [
23.474,
39.424,
47.464,
51.794,
54.636,
56.834,
58.228,
59.312,
60.186,
60.916,
61.942,
62.652,
63.43,
64.268,
65.248,
66.004,
66.904,
67.568,
68.076
]

vgg_HDQ_SWX_bpp = [
0.3168082099,
0.4289058575,
0.5277484653,
0.6175348231,
0.7018087895,
0.7795436036,
0.8478009118,
0.9152924465,
0.9776802941,
1.043899722,
1.121765869,
1.215418604,
1.331151254,
1.479714901,
1.680314702,
1.970697948,
2.440516455,
3.393653566,
8.493884778
]


plt.hlines(y=69.02, xmin=0, xmax=max(vgg_HDQ_SWX_bpp), label="Top-1 Acc",alpha=0.5)
plt.plot(vgg_jpeg_bpp,vgg_jpeg_acc,label="HDQ", marker=".", linestyle= '--')
plt.plot(vgg_HDQ_SWX_bpp , vgg_HDQ_SWX_acc ,label="SDQ_SWX", marker=".")

plt.title("VGG11")
plt.xlabel("rate(bpp)")
plt.ylabel("accuracy(%)")
plt.legend()
plt.grid(linestyle='--', linewidth=0.5)
plt.savefig("VGG11_SDQ_SWX_420.png",dpi=600)