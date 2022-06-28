# # for QF_YC in `seq 15 5 100`
# for QF_YC in 100
# # do
# # 	export file=./Resize_Compress/HDQ_VGG11_QF${QF_YC}_YUV.txt
# # 	# export file=HDQ_VGG11_QF${QF_YC}_SWX.txt
# # 	echo ${file}
# # 	python3 HDQtest_resize_compress.py --Model VGG11 --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} >> ${file}
# # done
# do
# 	export file=./Compress_Resize/HDQ_VGG11_QF${QF_YC}_YUV.txt
# 	# export file=HDQ_VGG11_QF${QF_YC}_SWX.txt
# 	echo ${file}
# 	python3 HDQtest_compress_resize.py --Model VGG11 --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} >> ${file}
	
# done
python3 HDQtest.py --Model Squeezenet --J 4 --a 4 --b 4 --QF_Y 100 --QF_C 100
