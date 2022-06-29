for model in Alexnet
do
	for QF_YC in `seq 10 5 100`
	# for QF_YC in 100
	do
		export file=./Resize_Compress/HDQ/${model}/${model}_QF${QF_YC}_YUV.txt
		# export file=./Compress_Resize/HDQ/${model}/${model}_QF${QF_YC}_YUV.txt
		echo ${file}
		python3 HDQtest_dataloader.py --Model ${model} --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} -resize_compress >> ${file}
	done
done

