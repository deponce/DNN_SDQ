# g++ -std=c++11 HDQmain.cpp -o HDQ_output $(pkg-config opencv4 --cflags --libs) -lpthread
g++ -std=c++11 HDQ_OptD_main.cpp -o HDQ_OptD_main $(pkg-config opencv4 --cflags --libs)

# for Opt-D 
# ./HDQ_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 50 -q 50 -B 1 -L 0   # done
# echo

export q_max=$1
export DT_Y=$2
export d_Y=$3
export BPP_t=$4
export QF_Y=$5
export QF_C=$6

./HDQ_OptD_main -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -T ${DT_Y} -t ${DT_Y} -D ${d_Y} -d ${d_Y} -X ${q_max} -x ${q_max} -U ${BPP_t} # done
echo


# BPP = 0.25
# QMAX_Y = 54
# DT_Y = 146839	d_waterLevel_Y = 128644 
# BPP: 0.254009
# PSNR: 31.9704

# BPP = 0.5
# DT_Y = 1521	d_waterLevel_Y = 34.7368
# BPP: 0.51598
# PSNR: 35.5831

# BPP = 0.75 // 37.4003
# DT_Y = 998	d_waterLevel_Y = 18.6795
# BPP: 0.754955
# PSNR: 37.4003

# BPP = 1
# DT_Y = 682	d_waterLevel_Y = 11.2434
# BPP: 1.02975
# PSNR: 39.0055

# BPP = 1.25
# DT_Y = 550	d_waterLevel_Y = 8.67612
# BPP: 1.25806
# PSNR: 40.0423

# BPP = 1.5
# DT_Y = 430	d_waterLevel_Y = 6.63497
# BPP: 1.52564
# PSNR: 41.2691

# BPP = 1.75
# DT_Y = 332	d_waterLevel_Y = 5.10769
# BPP: 1.77871
# PSNR: 42.3755

# BPP = 2 // 43.591
# DT_Y = 258	d_waterLevel_Y = 3.96924
# BPP: 2.05003
# PSNR: 43.591


# //////////////////////////////////////////////////


#  for Vanilla HDQ
# g++ -std=c++11 HDQmain.cpp -o HDQmain $(pkg-config opencv4 --cflags --libs)
# ./HDQ_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 14 -q 50 -B 1 -L 0   # done
# echo
# ./HDQ_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 15 -q 50 -B 1 -L 0   # done
# echo
# echo
# ./HDQ_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 39 -q 50 -B 1 -L 15 # done
# echo
# ./HDQ_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 40.5 -q 50 -B 1 -L 15 # done
# echo
# echo
# ./HDQ_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 63 -q 50 -B 1 -L 0   # done
# echo
# ./HDQ_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 64 -q 50 -B 1 -L 0   # done
# echo
# echo
# ./HDQ_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 76 -q 50 -B 1 -L 0   # done
# echo
# ./HDQ_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 77 -q 50 -B 1 -L 0   # done
# echo
