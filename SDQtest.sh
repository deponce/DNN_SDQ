for model in VGG11
do
	# for QF_YC in `seq 15 5 100`
	for QF_YC in 25
	do
		for beta in 10e5
		do
			export file=./Resize_Compress/SDQ/${model}/${model}_QF${QF_YC}_B${beta}_YUV.txt
			echo ${file}
			python3 SDQtest.py --Model ${model} --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} --Beta_S ${beta} --Beta_W ${beta} --Beta_X ${beta}  --L 1 >> ${file}
		done	
	done
done

