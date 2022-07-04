export root="/home/h2amer/work/workspace/ML_TS/"

# Resize then Compress
for model in VGG11
do
	# for QF_YC in `seq 10 5 100`
	for QF_YC in 10
	do
		# export file=./Resize_Compress/HDQ/SWX/${model}/${model}_QF${QF_YC}_SWX.txt
		export file=./Resize_Compress/HDQ/YUV/${model}/${model}_QF${QF_YC}_YUV.txt
		echo ${file}
		python3 HDQtest_dataloader.py --Model ${model} --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} \
									  -resize_compress --output_txt ${file} --device "cuda:0"  --root ${root}
	done
done

# --> Compress_Resize
# for model in Alexnet Squeezenet Resnet18
# do
# 	for QF_YC in `seq 100 -5 10`
# 	# for QF_YC in 100
# 	do
# 		# export file=./Compress_Resize/YUV/HDQ/${model}/${model}_QF${QF_YC}_YUV.txt
# 		export file=./Compress_Resize/HDQ/SWX444/${model}/${model}_QF${QF_YC}_SWX.txt
# 		echo ${file}
# 				python3 HDQtest_dataloader.py --Model ${model} --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} \
# 									  --output_txt ${file} --device "cuda:0" --root ${root}
# 	done
# done


# for model in Alexnet Squeezenet Resnet18
# do
# 	for QF_YC in `seq 100 -5 10`
# 	# for QF_YC in 100
# 	do
# 		# export file=./Compress_Resize/YUV/HDQ/${model}/${model}_QF${QF_YC}_YUV.txt
# 		export file=./Compress_Resize/HDQ/SWX420/${model}/${model}_QF${QF_YC}_SWX.txt
# 		echo ${file}
# 				python3 HDQtest_dataloader.py --Model ${model} --J 4 --a 2 --b 0 --QF_Y ${QF_YC} --QF_C ${QF_YC} \
# 									  --output_txt ${file} --device "cuda:0" --root ${root}
# 	done
# done


