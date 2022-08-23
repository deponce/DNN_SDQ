# export root="/home/h2amer/AhmedH.Salamah/ilsvrc2012"
export root="/home/h2amer/work/workspace/ML_TS/"
# export root="~/data/ImageNet/2012"

# export sens=SenMap_Scale_Norm
# export sens=SenMap_Normalized

# export addText="_wo_DC"

# export min_beta=0.5




# for model in Resnet18 
# do
# 	# for QF_YC in `seq 90 -10 80`
# 	# for QF_YC in `seq 70 -5 30`
# 	for QF_YC in 100
# 	do
# 		for beta in 30 10 40 50
# 		# for beta in 20 10 7.5 5 3 1 
# 		do
# 			echo "Beta : "${beta}

# 			export file=./Resize_Compress/SDQ/${model}/${model}_QF${QF_YC}_B${beta}_${sens}.txt
# 			export sens_dir=./SenMap_All/${sens}/${model}
# 			echo ${file}
# 			python3 SDQtest_dataloader.py --Model ${model} --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} --Beta_S ${beta} --Beta_W ${beta} --Beta_X ${beta}  --L 1.0 \
# 					-resize_compress  --colorspace 0 \
# 					--output_txt ${file} --device "cuda:1" --root ${root} --SenMap_dir ${sens_dir} 
# 		done	
# 	done
# done

# export colorspace=0
# export sens=SenMap_Normalized
# # export sens=NoModel

# for model in Resnet18
# do
# 		export file=./Resize_Compress/SDQ_OptD/${model}/YUV/${model}_B%.2f_sens_${sens}_d_water_Y%.4f_d_water_C%.4f_Q_max_Y%d_Q_max_C%d.txt
# 		export sens_dir=./SenMap_All/${sens}/${model}
# 		# echo ${file}
# 		python3 SDQ_OptD_dataloader.py --Model ${model} --J 4 --a 4 --b 4 \
# 									  -resize_compress --colorspace ${colorspace} \
# 									  --Qmax_Y 46 --Qmax_C 46 --DT_Y 100 --DT_C 100 \
# 									  --d_waterlevel_Y 0 --d_waterlevel_C 0  \
# 									  --Beta_S 100 --Beta_W 100 --Beta_X 100 --L 1.0 \
# 									  --output_txt ${file} --device "cuda:0" --root ${root} \
# 									  --SenMap_dir ${sens_dir}  --OptD True
# done


# for model in  VGG11 
# do
# 	# for QF_YC in `seq 90 -10 80`
# 	for QF_YC in `seq 70 -5 30`
# 	do
# 		# for beta in 7.5 5 2.5 1 0.75 0.5 0.25 0.1
# 		for min_beta in 1 0.75 0.5 
# 		do
# 			beta=$(echo "scale = 2; $QF_YC * 0.5+ $min_beta" | bc)
# 			beta=${beta%.*}
# 			echo "Beta : "${beta}

# 			export file=./Resize_Compress/SDQ/${model}/${model}_QF${QF_YC}_B${beta}_${sens}${addText}.txt
# 			export sens_dir=./SenMap_All/${sens}/${model}
# 			echo ${file}
# 		# 	python3 SDQtest_dataloader.py --Model ${model} --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} --Beta_S ${beta} --Beta_W ${beta} --Beta_X ${beta}  --L 1.0 \
# 		# 			-resize_compress  --colorspace 0 \
# 		# 			--output_txt ${file} --device "cuda:0" --root ${root} --SenMap_dir ${sens_dir} 
# 		done	
# 	done
# done


# export beta=1

# # Resize then Compress  senstivity
# for model in VGG11
# do
# 	for QF_YC in 80
# 	# for QF_YC in 70
# 	do
# 		# for sens in "SenMap_Org" "SenMap_Resize_Normalized" "SenMap_Scale_Normalized1" "SenMap_Scale_Normalized2" "SenMap_Scale_Normalized3" "SenMap_Scale"
# 		for sens in "SenMap_Resize_Normalized" "SenMap_Scale_Normalized1" "SenMap_Scale_Normalized2" "SenMap_Scale_Normalized3" "SenMap_Scale"
# 		do
# 			export file=./Resize_Compress/SDQ/${model}/${model}_QF${QF_YC}_B${beta}_${sens}.txt
# 			export sens_dir=./SenMap_All/${sens}/${model}
# 			echo ${file}
# 			python3 SDQtest_dataloader.py --Model ${model} --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} --Beta_S ${beta} --Beta_W ${beta} --Beta_X ${beta}  --L 1 \
# 					-resize_compress  --colorspace 0 \
# 					--output_txt ${file} --device "cuda:0" --root ${root} --SenMap_dir ${sens_dir} 
# 		done	
# 	done
# done


export colorspace=0
export sens=SenMap_Normalized
# export sens=NoModel

for model in Squeezenet
do
		export file=./Resize_Compress/SDQ_OptD/${model}/YUV/${model}_B%.2f_sens_${sens}_d_water_Y%.4f_d_water_C%.4f_Q_max_Y%d_Q_max_C%d.txt
		export sens_dir=./SenMap_All/${sens}/${model}
		# echo ${file}
		python3 SDQ_OptD_dataloader.py --Model ${model} --J 4 --a 4 --b 4 \
									  -resize_compress --colorspace ${colorspace} \
									  --Qmax_Y 46 --Qmax_C 46 --DT_Y 100 --DT_C 100 \
									  --d_waterlevel_Y 0 --d_waterlevel_C 0  \
									  --Beta_S 100 --Beta_W 100 --Beta_X 100 --L 1.0 \
									  --output_txt ${file} --device "cuda:1" --root ${root} \
									  --SenMap_dir ${sens_dir}  --OptD True
done