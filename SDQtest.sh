# export root="/home/h2amer/AhmedH.Salamah/ilsvrc2012"
export root="/home/h2amer/work/workspace/ML_TS/"
# export root="~/data/ImageNet/2012"


export beta=1
# Resize then Compress [ Deafult SDQ ]
for model in NoModel
do
	for QF_YC in 80
	do
		for sens in "SenMap_Org"
		do
			export file=./Resize_Compress/SDQ/${model}/${model}_QF${QF_YC}_B${beta}_${sens}.txt
			export sens_dir=./SenMap_All/${sens}/
			echo ${file}
			python3 SDQtest_dataloader.py --Model ${model} --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} --Beta_S ${beta} --Beta_W ${beta} --Beta_X ${beta}  --L 30 \
					-resize_compress --output_txt ${file} --device "cuda:0" --root ${root} --SenMap_dir ${sens_dir} 
		done	
	done
done

export beta=10e5

# Resize then Compress
for model in Squeezenet
do
	for QF_YC in 80
	# for QF_YC in 70
	do
		# for sens in "SenMap_Org" "SenMap_Resize_Normalized" "SenMap_Scale_Normalized1" "SenMap_Scale_Normalized2" "SenMap_Scale_Normalized3" "SenMap_Scale"
		for sens in "SenMap_Scale_Normalized2" "SenMap_Scale_Normalized3" "SenMap_Scale"
		do
			export file=./Resize_Compress/SDQ/${model}/${model}_QF${QF_YC}_B${beta}_${sens}.txt
			export sens_dir=./SenMap_All/${sens}/
			echo ${file}
			python3 SDQtest_dataloader.py --Model ${model} --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} --Beta_S ${beta} --Beta_W ${beta} --Beta_X ${beta}  --L 1 \
					-resize_compress --output_txt ${file} --device "cuda:0" --root ${root} --SenMap_dir ${sens_dir} 
		done	
	done
done


# for model in VGG11
# do
# 	for QF_YC in `seq 100 -5 0`
# 	# for QF_YC in 70
# 	do
# 		for beta in 10e5
# 		do
# 			export file=./Resize_Compress/SDQ/${model}/${model}_QF${QF_YC}_B${beta}_YUV.txt
# 			echo ${file}
# 			python3 SDQtest_dataloader.py --Model ${model} --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} --Beta_S ${beta} --Beta_W ${beta} --Beta_X ${beta}  --L 1 \
# 					-resize_compress --output_txt ${file} --device "cuda:1"	--root ${root}
# 		done	
# 	done
# done


# Compress then resize 
# for model in Squeezenet
# do
# 	# for QF_YC in `seq 70 -5 10`
# 	for QF_YC in 10
# 	do
# 		for beta in 10e4
# 		do
# 			export file=./Compress_Resize/SDQ/${model}/${model}_QF${QF_YC}_B${beta}_YUV.txt
# 			echo ${file}
# 			python3 SDQtest_dataloader.py --Model ${model} --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} --Beta_S ${beta} --Beta_W ${beta} --Beta_X ${beta}  --L 1 \
# 					--output_txt ${file} --device "cuda:1" --root ${root}
# 		done	
# 	done
# done

