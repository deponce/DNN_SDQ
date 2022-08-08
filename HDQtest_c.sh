export root=/home/h2amer/work/workspace/ML_TS/
# export root="~/data/ImageNet/2012"
# export root="/home/h2amer/AhmedH.Salamah/ilsvrc2012"


# export DT_Y=10
# export DT_C=10

# export Qmax_Y=255
# export Qmax_C=255

# export colorspace=0
# export sens=SenMap_Normalized
# # export sens=NoModel

# # Resize then Compress
# for model in Resnet18 Squeezenet
# do
# 	# for i in "${!DT_Y[@]}"; do
# 	# for i in "${!Qmax_Y[@]}"; do
# 	for d_waterlevel in `seq 1 1 4`; do
# 		d_waterlevel_Y=$(echo "scale = 2; $d_waterlevel / 100" | bc)
# 		d_waterlevel_C=$(echo "scale = 2; $d_waterlevel_Y * 2" | bc)
# 		echo "d_waterlevel_Y:"${d_waterlevel_Y}" ... d_waterlevel_C:"${d_waterlevel_C}
# 		export file=./Resize_Compress/HDQ_OptD/${model}/YUV/${model}_sens_${sens}_d_water_Y${d_waterlevel_Y}_Q_max_Y${Qmax_Y}.txt
# 		export sens_dir=./SenMap_All/${sens}/${model}
# 		echo ${file}
# 		time python3 HDQ_OptD_dataloader.py --Model ${model} --J 4 --a 4 --b 4 \
# 									  -resize_compress --colorspace ${colorspace} \
# 									  --Qmax_Y ${Qmax_Y}  --Qmax_C ${Qmax_C} --DT_Y ${DT_Y} --DT_C ${DT_C} \
# 									  --d_waterlevel_Y ${d_waterlevel_Y}  --d_waterlevel_C ${d_waterlevel_C}  \
# 									  --output_txt ${file} --device "cuda:1" --root ${root} \
# 									  --SenMap_dir ${sens_dir}  

# 	done
# done

# d_waterlevel_Y=('0.04' '0.04' '0.04' '0.04' '0.04' '0.04' '0.04' '0.04' '0.04' '0.04' '0.04' '0.04' '0.04' '0.04' '0.04' '0.04' '0.04' '0.04' '0.04')
# d_waterlevel_C=('0.02' '0.03' '0.04' '0.05' '0.06' '0.07' '0.08' '0.09' '0.10' '0.11' '0.12' '0.13' '0.14' '0.15' '0.16' '0.17' '0.18' '0.19' '0.20')

# Qmax_Y=('46')
# Qmax_C=('46')



# Resize then Compress

# for mult in 10000; do
# 	for model in Squeezenet
# 	do
# 		# for i in "${!DT_Y[@]}"; do
# 		for i in "${!d_waterlevel_Y[@]}"; do
# 			d_waterlevel_Y_=$(echo "scale = 2; ${d_waterlevel_Y[i]} * ${mult}" | bc)
# 			d_waterlevel_C_=$(echo "scale = 2; ${d_waterlevel_C[i]} * ${mult}" | bc)
# 			# export file=./Resize_Compress/HDQ/SWX444/${model}/${model}_QF${QF_YC}_SWX_${colorspace}.txt
# 			export file=./Resize_Compress/HDQ_OptD_correct_YUV/${model}/YUV/${model}_sens_${sens}_d_water_C${d_waterlevel_C_}_Q_max_Y${Qmax_Y}.txt
# 			printf '${DT_Y[%s]}=%s\t${DT_C[%s]}=%s\n' "$i" "${DT_Y[i]}" "$i" "${DT_C[i]}"
# 			export sens_dir=./SenMap_All/${sens}/${model}
# 			echo ${file}
# 			python3 HDQ_OptD_dataloader.py --Model ${model} --J 4 --a 4 --b 4 \
# 										  -resize_compress --colorspace ${colorspace} \
# 										  --Qmax_Y 46 --Qmax_C 46 --DT_Y 1 --DT_C 1 \
# 										  --d_waterlevel_Y ${d_waterlevel_Y_}  --d_waterlevel_C ${d_waterlevel_C_}  \
# 										  --output_txt ${file} --device "cuda:1" --root ${root} \
# 										  --SenMap_dir ${sens_dir}  

# 		done
# 	done
# done

# for mult in 10 100 1000 10000; do
# 	for model in Resnet18 VGG11 Alexnet
# 	do
# 		# for i in "${!DT_Y[@]}"; do
# 		for i in "${!d_waterlevel_Y[@]}"; do
# 			d_waterlevel_Y_=$(echo "scale = 2; ${d_waterlevel_Y[i]} * ${mult}" | bc)
# 			d_waterlevel_C_=$(echo "scale = 2; ${d_waterlevel_C[i]} * ${mult}" | bc)
# 			# export file=./Resize_Compress/HDQ/SWX444/${model}/${model}_QF${QF_YC}_SWX_${colorspace}.txt
# 			export file=./Resize_Compress/HDQ_OptD_correct_YUV/${model}/YUV/${model}_sens_${sens}_d_water_Y${d_waterlevel_Y_}_d_water_C${d_waterlevel_C_}_Q_max_Y46.txt
# 			# printf '${DT_Y[%s]}=%s\t${DT_C[%s]}=%s\n' "$i" "${DT_Y[i]}" "$i" "${DT_C[i]}"
# 			export sens_dir=./SenMap_All/${sens}/${model}
# 			echo ${file}
# 			python3 HDQ_OptD_dataloader.py --Model ${model} --J 4 --a 4 --b 4 \
# 										  -resize_compress --colorspace ${colorspace} \
# 										  --Qmax_Y 46 --Qmax_C 46 --DT_Y 1 --DT_C 1 \
# 										  --d_waterlevel_Y ${d_waterlevel_Y_}  --d_waterlevel_C ${d_waterlevel_C_}  \
# 										  --output_txt ${file} --device "cuda:1" --root ${root} \
# 										  --SenMap_dir ${sens_dir}  

# 		done
# 	done
# done


export colorspace=0
# export sens=SenMap_Normalized
export sens=NoModel

for model in Alexnet Resnet18 VGG11 Squeezenet
do
		export file=./Resize_Compress/HDQ_OptD_correct_YUV_Qmax/${model}/YUV/${model}_sens_${sens}_d_water_Y%.2f_d_water_C%.2f_Q_max_Y%d_Q_max_C%d.txt
		export sens_dir=./SenMap_All/${sens}/${model}
		# echo ${file}
		python3 HDQ_OptD_dataloader.py --Model ${model} --J 4 --a 4 --b 4 \
									  -resize_compress --colorspace ${colorspace} \
									  --Qmax_Y 46 --Qmax_C 46 --DT_Y 1 --DT_C 1 \
									  --d_waterlevel_Y 0 --d_waterlevel_C 0  \
									  --output_txt ${file} --device "cuda:1" --root ${root} \
									  --SenMap_dir ${sens_dir}  

done
