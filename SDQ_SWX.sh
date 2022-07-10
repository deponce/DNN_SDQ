export root="/home/h2amer/AhmedH.Salamah/ilsvrc2012"
# export root="/home/h2amer/work/workspace/ML_TS/"
# export root="~/data/ImageNet/2012"


export beta=1
export QF_YC=80
export sens="NoModel"

# Resize then Compress [ Deafult SDQ with SWX] No senstivity
for model in VGG11
do
	for lamda in 10
	do
		# for QF_YC in 60
		for QF_YC in `seq 100 -10 40`
		do
			for colorspace in 0 1
			do
				export file=./Resize_Compress/SDQ/${model}/${model}_QF${QF_YC}_L${lamda}_${sens}_colorspace${colorspace}.txt
				export sens_dir=./SenMap_All/${sens}/${model}
				echo ${file}
				python3 SDQtest_dataloader.py --Model ${model} --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} --Beta_S ${beta} --Beta_W ${beta} --Beta_X ${beta}  --L ${lamda} \
						-resize_compress  --colorspace ${colorspace} \
						--output_txt ${file} --device "cuda:0" --root ${root} --SenMap_dir ${sens_dir} 
			done	
		done
	done
done


# for model in VGG11
# do
# 	for lamda in 30 
# 	do
# 		# for QF_YC in `seq 100 -10 10`
# 		for QF_YC in 100
# 		do
# 			for colorspace in 0 1
# 			do
# 				export file=./Resize_Compress/SDQ/${model}/${model}_QF${QF_YC}_L${lamda}_${sens}_colorspace${colorspace}.txt
# 				export sens_dir=./SenMap_All/${sens}/${model}
# 				echo ${file}
# 				python3 SDQtest_dataloader.py --Model ${model} --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} --Beta_S ${beta} --Beta_W ${beta} --Beta_X ${beta}  --L ${lamda} \
# 						-resize_compress  --colorspace ${colorspace} \
# 						--output_txt ${file} --device "cuda:0" --root ${root} --SenMap_dir ${sens_dir} 
# 			done	
# 		done
# 	done
# done


# export QF_YC = 80
# export sens="SenMap_Resize_Normalized"
# export beta=1

# # Resize then Compress
# for model in VGG11
# do
# 	for do colorspace in 1 2
# 	# for QF_YC in 70
# 	do
# 		export file=./Resize_Compress/SDQ/${model}/${model}_QF${QF_YC}_B${beta}_${sens}_colorspace${colorspace}.txt
# 		export sens_dir=./SenMap_All/${sens}/
# 		echo ${file}
# 		python3 SDQtest_dataloader.py --Model ${model} --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} --Beta_S ${beta} --Beta_W ${beta} --Beta_X ${beta}  --L 1 \
# 				-resize_compress  --colorspace 0 \
# 				--output_txt ${file} --device "cuda:0" --root ${root} --SenMap_dir ${sens_dir} 
# 	done
# done