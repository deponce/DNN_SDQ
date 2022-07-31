# g++ -std=c++11 HDQmain.cpp -o HDQ_output $(pkg-config opencv4 --cflags --libs) -lpthread
g++ -std=c++11 HDQmain.cpp -o HDQ_output $(pkg-config opencv4 --cflags --libs)

# for Opt-D 
# ./HDQ_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 50 -q 50 -B 1 -L 0   # done
# echo

export d_Y=$1
export q_max=$2
./HDQ_output -M NoModel -P ./sample/lena223.tif -J 4 -a 4 -b 4 -Q 50 -q 50 -D ${d_Y} -d ${d_Y} -X ${q_max}  -x ${q_max}  # done
echo


#  for Vanilla HDQ

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
