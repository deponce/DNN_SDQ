g++ -std=c++11 SDQmain.cpp -o SDQoutput $(pkg-config opencv4 --cflags --libs) -lpthread 


export q_max=$1
export DT_Y=$2
export d_Y=$3
export lambda=$4

# d --> 36 // L --> 50
# ./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 50 -q 50 -B 1 -L ${lambda} -D ${d_Y} -d ${d_Y} -X ${q_max} -x ${q_max}  # done
# echo

# d --> 36
# ./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 50 -q 50 -B 1 -L ${lambda} -D ${d_Y} -d ${d_Y} -X ${q_max} -x ${q_max}  # done
# echo

# DT --> 260 (d=4) // d --> -1 // L --> 1.5
./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 50 -q 50 -B 1 -L ${lambda} -T ${DT_Y} -t ${DT_Y} -D ${d_Y} -d ${d_Y} -X ${q_max} -x ${q_max}  # done
echo

# ./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 13 -q 50 -B 1 -L 60  # done
# echo
# ./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 38 -q 50 -B 1 -L 20 # done
# echo
# ./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 65 -q 50 -B 1 -L 13 # done
# echo
# ./SDQoutput -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 78 -q 50 -B 1 -L 5 # done

# ILSVRC2012_val_00017916.JPEG  lena3.tif
