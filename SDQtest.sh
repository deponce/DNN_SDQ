for QF_YC in 80 25
do
	for beta in 300
	do
		export file=./Compress_Resize/SDQ_VGG11_QF${QF_YC}_B${beta}_YUV.txt
		echo ${file}
		python3 SDQtest.py --Model VGG11 --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} --Beta_S ${beta} --Beta_W ${beta} --Beta_X ${beta}  --L 1 >> ${file}
	done	
done