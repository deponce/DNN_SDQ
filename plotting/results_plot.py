import numpy as np
import matplotlib.pyplot as plt


SDQ_acc= [
44.432,
58.596,
62.678,
64.23,
64.362,
64.578,
65.046,
65.322,
65.478,
65.734,
66.03,
66.372,
66.05,
66.616,
67.044,
66.224,
66.988,
67.504,
65.838,
67.976,
63.818,
65.6,
67.358,
]

SDQ_bpp=[
0.4129337863,
0.6402428947,
0.8299257332,
0.9638630792,
0.9770054287,
0.9903201861,
1.079388675,
1.098559429,
1.118088631,
1.22455697,
1.254720472,
1.285410294,
1.393180182,
1.443150623,
1.494850695,
1.630139532,
1.721899222,
1.825344672,
1.963265353,
2.378258268,
3.311368215,
3.599311,
4.131443199,
]


vgg_jpeg_acc = [33.282, 48.716, 54.762, 60.49, 62.736, 64.004, 65.352, 66.354, 67.41, 68.412, 68.884, 68.956, 68.986]
vgg_jpeg_bpp = [0.3822962079, 0.5091492985, 0.6160215632, 0.821426690715998, 0.987924535, 1.131426451, 1.327681325, 1.586540155, 2.024759671, 2.96028335, 5.382353456, 6.507102903, 10.10468878]


def maxRate(acc, bpp, th):
    new_acc = []
    new_bpp = []
    for idx, b in enumerate(bpp):
        if b > th:
            new_acc.append(acc[idx])
            new_bpp.append(bpp[idx])
    return new_acc, new_bpp

def maxAcc(acc, bpp, th):
    new_acc = []
    new_bpp = []
    for idx, Acc in enumerate(acc):
        if Acc > th:
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


# VGG11

min_rate = 60

vgg_qf10_acc = [ 44.48, 44.47, 44.59, 44.69, 44.69, 44.71, 43.60, 44.76, 44.75 ]
vgg_qf10_bpp = [
0.4119690306,
0.4130215107,
0.4134972066,
0.4142955344,
0.4144311961,
0.4145250497,
0.4133868283,
0.4145745133,
0.4145897884,
]

vgg_qf50_acc = [56.93, 65.26, 65.50, 65.59, 65.55, 65.62]
vgg_qf50_bpp = [0.8717956941, 1.08562427, 1.113521173, 1.130422168, 1.133574012, 1.135881871]

vgg_qf60_acc = [55.73, 65.84, 66.324, 66.438, 66.402, 66.38]
vgg_qf60_bpp = [0.9356502687, 1.233955427, 1.277982205, 1.304607234, 1.309606448, 1.313290951]


vgg_qf70_acc = [30.63, 54.29, 64.81, 66.39, 67.28]
vgg_qf70_bpp = [0.7822936323, 0.9985315958, 1.26752342, 1.40752254, 1.541578649]




vgg_qf80_acc=[51.586, 64.14, 66.59, 67.14 ,67.38 ,67.73 ,67.85 ,67.82 ,67.87 ,67.87 , 67.88, 67.87, 67.86, 67.89 ]
vgg_qf80_bpp=[1.08  , 1.43 , 1.65 , 1.75  ,1.80  ,1.88  ,1.91  ,1.92  ,1.92  ,1.92  , 1.92656210598051 , 1.92878 , 1.93, 1.931]

vgg_qf90_acc=[48.31, 62.86, 66.32, 67.33, 67.76, 68.27, 68.48, 68.50, 68.53, 68.53, 68.50, 68.51, 68.54, 68.54]
vgg_qf90_bpp=[1.189732928, 1.633789459, 1.984493891, 2.16509928,2.279440895,2.531676747,2.62438051,2.655996182,2.671729836,2.671933753, 2.687405263, 2.695223752, 2.699888647, 2.70300334 ]


vgg_qf100_acc = [48.10, 66.90, 68.84, 69.04, 68.99]
vgg_qf100_bpp = [1.820553424, 3.348418722, 5.187747278, 6.085728048, 6.319634313]

# SDQ_acc

criterial = maxAcc

vgg_qf10_acc, vgg_qf10_bpp = criterial(vgg_qf10_acc, vgg_qf10_bpp, min_rate)
vgg_qf50_acc, vgg_qf50_bpp = criterial(vgg_qf50_acc, vgg_qf50_bpp, min_rate)
vgg_qf60_acc, vgg_qf60_bpp = criterial(vgg_qf60_acc, vgg_qf60_bpp, min_rate)
vgg_qf70_acc, vgg_qf70_bpp = criterial(vgg_qf70_acc, vgg_qf70_bpp, min_rate)
vgg_qf80_acc, vgg_qf80_bpp = criterial(vgg_qf80_acc, vgg_qf80_bpp, min_rate)
vgg_qf90_acc, vgg_qf90_bpp = criterial(vgg_qf90_acc, vgg_qf90_bpp, min_rate)
vgg_qf100_acc, vgg_qf100_bpp = criterial(vgg_qf100_acc, vgg_qf100_bpp, min_rate)
vgg_jpeg_acc, vgg_jpeg_bpp = criterial(vgg_jpeg_acc, vgg_jpeg_bpp, min_rate)
SDQ_acc, SDQ_bpp = criterial(SDQ_acc, SDQ_bpp, min_rate)


# vgg_qf80_bpp = [(24 / x) for x in vgg_qf80_bpp]
# vgg_qf90_bpp = [(24 / x) for x in vgg_qf90_bpp]
# vgg_jpeg_bpp = [(24 / x) for x in vgg_jpeg_bpp]

# vgg_jpeg_acc = [ 66.354, 67.41, 68.412, 68.986]
# vgg_jpeg_bpp = [ 1.586540155, 2.024759671, 2.96028335, 10.10468878]

plt.plot(vgg_qf10_bpp,vgg_qf10_acc,label="SDQ_QF10", marker=".")
plt.plot(vgg_qf50_bpp,vgg_qf50_acc,label="SDQ_QF50", marker=".")
plt.plot(vgg_qf60_bpp,vgg_qf60_acc,label="SDQ_QF60", marker=".")
plt.plot(vgg_qf70_bpp,vgg_qf70_acc,label="SDQ_QF70", marker=".")
plt.plot(vgg_qf80_bpp,vgg_qf80_acc,label="SDQ_QF80", marker=".")
plt.plot(vgg_qf90_bpp,vgg_qf90_acc,label="SDQ_QF90", marker=".")
plt.plot(vgg_qf100_bpp,vgg_qf100_acc,label="SDQ_QF100", marker=".")
plt.plot(vgg_jpeg_bpp,vgg_jpeg_acc,label="HDQ", marker=".", linestyle= '--')
plt.scatter(SDQ_bpp,SDQ_acc,label="SDQ_YUV", marker=".",  c='pink')

plt.title("VGG11")
plt.xlabel("rate(bpp)")
plt.ylabel("accuracy(%)")
plt.legend()
# plt.show()
plt.savefig("VGG11.png",dpi=600)



# SWX vs JPEG

SDQ_SWX_acc = [
68.392,
68.986,
68.854,
68.768,
65.158,
65.188,
66.086,
66.032,
66.118,
66.454,
66.45,
66.46,
66.828,
66.81,
66.874,
67.214,
67.238,
67.204,
67.596,
67.544,
67.56,
68.754,
68.71,
68.688,
]

SDQ_SWX_bpp = [
7.011797518,
9.889757026,
8.561356108,
7.897722763,
1.212594566,
1.209628398,
1.388890924,
1.385438052,
1.384285475,
1.500577103,
1.499112356,
1.504968553,
1.643417489,
1.641512795,
1.649105745,
1.813771927,
1.824526741,
1.816457868,
2.064931344,
2.052877066,
2.048855125,
4.077217355,
3.946662501,
3.90582086,
]

HDQ_SWX_acc = [
27.862,
44.596,
52.35,
56.558,
58.93,
60.612,
61.848,
62.686,
63.3,
64.028,
64.688,
65.316,
65.854,
66.516,
67.192,
67.694,
68.34,
68.702,
69.008
]

HDQ_SWX_bpp = [
0.3850757139,
0.5154650346,
0.6342837216,
0.7448789272,
0.8508629177,
0.9503689111,
1.039074576,
1.127665517,
1.210340339,
1.298283244,
1.401957989,
1.526531158,
1.680283832,
1.877234094,
2.143470584,
2.5327036,
3.176413875,
4.547688356,
13.52919593
]


plt.figure()

plt.plot(vgg_jpeg_bpp,vgg_jpeg_acc,label="HDQ_YUV", marker=".", linestyle= '--')
plt.plot(HDQ_SWX_bpp,HDQ_SWX_acc,label="HDQ_SWX", marker=".", linestyle= '--')
plt.scatter(SDQ_SWX_bpp,SDQ_SWX_acc,label="SDQ_SWX", marker=".",  c='red')
# plt.scatter(SDQ_bpp,SDQ_acc,label="SDQ_YUV", marker=".",  c='pink')
plt.title("VGG11")
plt.xlabel("rate(bpp)")
plt.ylabel("accuracy(%)")
plt.legend()
# plt.show()