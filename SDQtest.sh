export root="/home/h2amer/AhmedH.Salamah/ilsvrc2012"
# export root="/home/h2amer/work/workspace/ML_TS/"
# export root="~/data/ImageNet/2012"

# export sens=SenMap_Scale_Norm
export sens=SenMap_Normalized

# export addText="_wo_DC"

for model in  VGG11 
do
	# for QF_YC in `seq 90 -10 80`
	for QF_YC in 80
	do
		# for beta in 7.5 5 2.5 1 0.75 0.5 0.25 0.1
		for beta in 15 20 25 30
		# for beta in 0.5 
		do
			export file=./Resize_Compress/SDQ/${model}/${model}_QF${QF_YC}_B${beta}_${sens}${addText}.txt
			export sens_dir=./SenMap_All/${sens}/${model}
			echo ${file}
			python3 SDQtest_dataloader.py --Model ${model} --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} --Beta_S ${beta} --Beta_W ${beta} --Beta_X ${beta}  --L 1.0 \
					-resize_compress  --colorspace 0 \
					--output_txt ${file} --device "cuda:0" --root ${root} --SenMap_dir ${sens_dir} 
		done	
	done
done


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


