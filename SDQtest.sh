# for QF_YC in 80 25
# do
# 	for beta in 300
# 	do
# 		export file=./Compress_Resize/SDQ_VGG11_QF${QF_YC}_B${beta}_YUV.txt
# 		echo ${file}
#		python3 SDQtest.py --Model VGG11 --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} --Beta_S ${beta} --Beta_W ${beta} --Beta_X ${beta}  --L 1 >> ${file}
# 	done	
# done
python3 SDQtest.py --Model Squeezenet --J 4 --a 4 --b 4 --QF_Y 100 --QF_C 100 --Beta_S 10000 --Beta_W 10000 --Beta_X 10000 --L 1
