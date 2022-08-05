g++ -std=c++11 SDQmain.cpp -o SDQoutput $(pkg-config opencv4 --cflags --libs) -lpthread 

# ./SDQ_CPP.sh 46 46 260 260 -1 -1 2 1.6

export q_max_Y=$1
export q_max_C=$2
export DT_Y=$3
export DT_C=$4
export d_Y=$5
export d_C=$6
export lambda=$7
export BPP_t=$8

# d --> 36 // L --> 50
# ./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 50 -q 50 -B 1 -L ${lambda} -D ${d_Y} -d ${d_Y} -X ${q_max} -x ${q_max}  # done
# echo

# d --> 36
# ./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 50 -q 50 -B 1 -L ${lambda} -D ${d_Y} -d ${d_Y} -X ${q_max} -x ${q_max}  # done
# echo

# DT --> 260 (d=4) // d --> -1 // L --> 1.5
./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 50 -q 50 -B 1 -L ${lambda} -T ${DT_Y} -t ${DT_C} -D ${d_Y} -d ${d_C} -X ${q_max_Y} -x ${q_max_C}  -U ${BPP_t}  # done


# BPP = 0.25 // 32.3776
# Qmax = 68
# DT_Y = 998	d_waterLevel_Y = 500
# Lmbda: 0.5
# BPP: 0.253674
# PSNR: 32.3776


# BPP = 0.5 // 36.3239
# Qmax = 46
# DT_Y = 998	d_waterLevel_Y = 50
# Lmbda: 0.5
# BPP: 0.510941
# PSNR: 36.3239



# BPP = 0.75 // 38.1244
# DT_Y = 998	d_waterLevel_Y = 18.6795	L = 11
# BPP: 0.75761
# PSNR: 38.1244


# BPP = 1 // 39.7221
# DT_Y = -10	d_waterLevel_Y = 12.5
# Lmbda: 0.5
# BPP: 1.08872
# PSNR: 39.7221



# BPP = 1.25 // 40.8236 --> repeat
# DT_Y = -10	d_waterLevel_Y = 8.5
# Lmbda: 4.5
# BPP: 1.26794
# PSNR: 40.8236



# BPP = 1.5 // 41.956
# DT_Y = 430	d_waterLevel_Y = 6.74315
# Lmbda: 2.7
# BPP: 1.54885
# PSNR: 41.956


# BPP = 1.75 // 43.0501
# DT_Y = 340	d_waterLevel_Y = 5.31251
# Lmbda: 2.2
# BPP: 1.77464
# PSNR: 43.0501


# BPP = 2 // 44.1512
# DT_Y = 258	d_waterLevel_Y = 3.96924	L = 1.5
# BPP: 2.03288
# PSNR: 44.1512

# //////////////////////////////////////////////////

#  for Vanilla SDQ

# ./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 13 -q 50 -B 1 -L 60  # done
# echo
# ./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 38 -q 50 -B 1 -L 20 # done
# echo
# ./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 65 -q 50 -B 1 -L 13 # done
# echo
# ./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 78 -q 50 -B 1 -L 5 # done

# ILSVRC2012_val_00017916.JPEG  lena3.tif
