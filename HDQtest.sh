# export root=/home/h2amer/work/workspace/ML_TS/
# export root="~/data/ImageNet/2012"
export root="/home/h2amer/AhmedH.Salamah/ilsvrc2012"


# Resize then Compress
# for model in Resnet18 Squeezenet 
# do
# 	# for QF_YC in 85
# 	for QF_YC in `seq 100 -5 10`
# 	do
# 		for colorspace in 0
# 		do
# 			# export file=./Resize_Compress/HDQ/SWX444/${model}/${model}_QF${QF_YC}_SWX_${colorspace}.txt
# 			export file=./Resize_Compress/HDQ/YUV444/${model}/${model}_QF${QF_YC}_YUV.txt
# 			echo ${file}
# 			python3 HDQtest_dataloader.py --Model ${model} --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} \
# 										  -resize_compress --colorspace ${colorspace} \
# 										  --output_txt ${file} --device "cuda:0" --root ${root}
# 		done
# 	done
# done


# Resize then Compress
# for model in Resnet18 Squeezenet 
# do
# 	# for QF_YC in 85
# 	for QF_YC in `seq 100 -5 10`
# 	do
# 		for colorspace in 0
# 		do
# 			# export file=./Resize_Compress/HDQ/SWX444/${model}/${model}_QF${QF_YC}_SWX_${colorspace}.txt
# 			export file=./Resize_Compress/HDQ/YUV420/${model}/${model}_QF${QF_YC}_YUV.txt
# 			echo ${file}
# 			python3 HDQtest_dataloader.py --Model ${model} --J 4 --a 2 --b 0 --QF_Y ${QF_YC} --QF_C ${QF_YC} \
# 										  -resize_compress --colorspace ${colorspace} \
# 										  --output_txt ${file} --device "cuda:0" --root ${root}
# 		done
# 	done
# done

# //////////////////////////////////////////

# DT_Y=('50' '150' '200' '300')
# DT_C=('80' '200' '300' '400')

# d_waterlevel_Y=('2' '5' '10' '15')
# d_waterlevel_C=('4' '10' '20' '30')

# d_waterlevel_Y=('0.1' )
# d_waterlevel_C=('0.2')

# Qmax_Y=('255')
# Qmax_C=('255')

# export colorspace=0
# export sens=SenMap_Normalized
# export sens=NoModel

# Resize then Compress
# for model in VGG11 
# do
# 	# for i in "${!DT_Y[@]}"; do
# 	# for i in "${!Qmax_Y[@]}"; do

# 		# export file=./Resize_Compress/HDQ/SWX444/${model}/${model}_QF${QF_YC}_SWX_${colorspace}.txt
# 		export file=./Resize_Compress/HDQ_OptD/${model}/YUV/${model}_d_water_Y${d_waterlevel_Y[i]}_Q_max_Y${Qmax_Y[i]}.txt
# 		printf '${DT_Y[%s]}=%s\t${DT_C[%s]}=%s\n' "$i" "${DT_Y[i]}" "$i" "${DT_C[i]}"
# 		export sens_dir=./SenMap_All/${sens}/${model}
# 		echo ${file}
# 		python3 HDQ_OptD_dataloader.py --Model ${model} --J 4 --a 4 --b 4 \
# 									  -resize_compress --colorspace ${colorspace} \
# 									  --Qmax_Y ${Qmax_Y[i]}  --Qmax_C ${Qmax_C[i]} --DT_Y ${DT_Y[i]} --DT_C ${DT_C[i]} \
# 									  --d_waterlevel_Y ${d_waterlevel_Y[i]}  --d_waterlevel_C ${d_waterlevel_C[i]}  \
# 									  --output_txt ${file} --device "cuda:0" --root ${root} \
# 									  --SenMap_dir ${sens_dir}  

# 	done
# done


# //////////////////////////////////////////

export DT_Y=10
export DT_C=10

export Qmax_Y=255
export Qmax_C=255

export colorspace=0
export sens=SenMap_Normalized
# export sens=NoModel

# Resize then Compress
for model in VGG11 Alexnet
do
	# for i in "${!DT_Y[@]}"; do
	# for i in "${!Qmax_Y[@]}"; do
	for d_waterlevel in `seq 5 5 1000`; do
		d_waterlevel_Y=$(echo "scale = 2; $d_waterlevel / 100" | bc)
		d_waterlevel_C=$(echo "scale = 2; $d_waterlevel_Y * 2" | bc)
		echo "d_waterlevel_Y:"${d_waterlevel_Y}" ... d_waterlevel_C:"${d_waterlevel_C}
		export file=./Resize_Compress/HDQ_OptD/${model}/YUV/${model}_sens_${sens}_d_water_Y${d_waterlevel_Y}_Q_max_Y${Qmax_Y}.txt
		export sens_dir=./SenMap_All/${sens}/${model}
		echo ${file}
		time python3 HDQ_OptD_dataloader.py --Model ${model} --J 4 --a 4 --b 4 \
									  -resize_compress --colorspace ${colorspace} \
									  --Qmax_Y ${Qmax_Y}  --Qmax_C ${Qmax_C} --DT_Y ${DT_Y} --DT_C ${DT_C} \
									  --d_waterlevel_Y ${d_waterlevel_Y}  --d_waterlevel_C ${d_waterlevel_C}  \
									  --output_txt ${file} --device "cuda:0" --root ${root} \
									  --SenMap_dir ${sens_dir}  

	done
done



# for model in VGG11 
# do
# 	# for QF_YC in `seq 100 -5 10`
# 	# do
# 	for i in "${!DT_Y[@]}"; do
# 		for colorspace in 2; do
# 			# export file=./Resize_Compress/HDQ/SWX444/${model}/${model}_QF${QF_YC}_SWX_${colorspace}.txt
# 			export file=./Resize_Compress/HDQ_OptD/${model}/SWX/${model}_DT_Y${DT_Y[i]}_DT_C${DT_C[i]}_SWX.txt
# 			printf '${DT_Y[%s]}=%s\t%s\n' "$i" "${DT_Y[i]}" "${DT_C[i]}"
# 			echo ${file}
# 			python3 HDQ_OptD_dataloader.py --Model ${model} --J 4 --a 4 --b 4 \
# 										  -resize_compress --colorspace ${colorspace} \
# 										  --Qmax_Y 46 --Qmax_C 46 --DT_Y ${DT_Y[i]} --DT_C ${DT_C[i]} \
# 										  --output_txt ${file} --device "cuda:0" --root ${root}
# 		done
# 	done
# done