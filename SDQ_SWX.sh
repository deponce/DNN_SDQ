export root="/home/h2amer/AhmedH.Salamah/ilsvrc2012"
# export root="/home/h2amer/work/workspace/ML_TS/"
# export root="~/data/ImageNet/2012"
# export root="/datashare/ImageNet/ILSVRC2012/validation/"


export beta=1
export QF_YC=70
export sens="NoModel"
export lamda=10
# export addText="_wo_DC"
# Resize then Compress [ Deafult SDQ with SWX] No senstivity
# for model in VGG11
# do
# 	for QF_YC in 100 95 80 75 70 65 60
# 	do
# 		for lamda in 10 
# 		# for QF_YC in `seq 100 -10 40`
# 		do
# 			for colorspace in 3
# 			do
# 				export file=./Resize_Compress/SDQ/${model}/${model}_QF${QF_YC}_L${lamda}_${sens}_colorspace${colorspace}${addText}.txt
# 				export sens_dir=./SenMap_All/${sens}/${model}
# 				echo ${file}
# 				python3 SDQtest_dataloader.py --Model ${model} --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} --Beta_S ${beta} --Beta_W ${beta} --Beta_X ${beta}  --L ${lamda} \
# 						-resize_compress  --colorspace ${colorspace} \
# 						--output_txt ${file} --device "cuda:0" --root ${root} --SenMap_dir ${sens_dir} 
# 			done	
# 		done
# 	done
# done


# for model in VGG11
# do
# 	# for slop in 25 29
# 	# for QF_YC in 95 80 75 70 65 60 50
# 	for QF_YC in 20
# 	do
# 		for lamda in 1
# 		do
# 			# lamda=$(echo "scale = 2; $QF_YC * -0.25 + $slop" | bc)
# 			lamda=${lamda%.*}
# 			echo "Lamdha : "${lamda}
# 			for colorspace in 0
# 			do
# 				export file=./Resize_Compress/SDQ/${model}/${model}_QF${QF_YC}_L${lamda}_${sens}_colorspace${colorspace}${addText}.txt
# 				export sens_dir=./SenMap_All/${sens}/${model}
# 				echo ${file}
# 				python3 SDQtest_dataloader_4models.py --Model ${model} --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} --Beta_S ${beta} --Beta_W ${beta} --Beta_X ${beta}  --L ${lamda} \
# 						-resize_compress  --colorspace ${colorspace} \
# 						--output_txt ${file} --device "cuda:1" --root ${root} --SenMap_dir ${sens_dir} 
# 			done
# 		done
# 	done
# done


# export addText="_Subt_meanPerImage"
export addText=""
for model in VGG11
do
	# for QF_YC in `seq 100 -5 10`
	for QF_YC in 100
	do
			for colorspace in 3
		do
			export file=./Resize_Compress/HDQ/SWX/${model}/${model}_QF${QF_YC}_${sens}_colorspace${colorspace}${addText}.txt
			# export file=./Resize_Compress/HDQ/YUV/${model}/${model}_QF${QF_YC}_YUV.txt
			echo ${file}
			python3 HDQtest_dataloader.py --Model ${model} --J 4 --a 4 --b 4 --QF_Y ${QF_YC} --QF_C ${QF_YC} \
										  -resize_compress --colorspace ${colorspace} \
										  --output_txt ${file} --device "cuda:1"  --root ${root}
		done
	done
done