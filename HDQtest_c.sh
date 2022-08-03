# export root=/home/h2amer/work/workspace/ML_TS/
# export root="~/data/ImageNet/2012"
export root="/home/h2amer/AhmedH.Salamah/ilsvrc2012"


export DT_Y=10
export DT_C=10

export Qmax_Y=255
export Qmax_C=255

export colorspace=0
# export sens=SenMap_Normalized
export sens=NoModel

# Resize then Compress
for model in VGG11 Alexnet
do
	# for i in "${!DT_Y[@]}"; do
	# for i in "${!Qmax_Y[@]}"; do
	for d_waterlevel in `seq 5 5 1000`; do
		d_waterlevel_Y=$(echo "scale = 2; $d_waterlevel / 10" | bc)
		d_waterlevel_C=$(echo "scale = 2; $d_waterlevel_Y * 2" | bc)
		echo "d_waterlevel_Y:"${d_waterlevel_Y}" ... d_waterlevel_C:"${d_waterlevel_C}
		export file=./Resize_Compress/HDQ_OptD/${model}/YUV/${model}_sens_${sens}_d_water_Y${d_waterlevel_Y}_Q_max_Y${Qmax_Y}.txt
		export sens_dir=./SenMap_All/${sens}/${model}
		echo ${file}
		time python3 HDQ_OptD_dataloader.py --Model ${model} --J 4 --a 4 --b 4 \
									  -resize_compress --colorspace ${colorspace} \
									  --Qmax_Y ${Qmax_Y}  --Qmax_C ${Qmax_C} --DT_Y ${DT_Y} --DT_C ${DT_C} \
									  --d_waterlevel_Y ${d_waterlevel_Y}  --d_waterlevel_C ${d_waterlevel_C}  \
									  --output_txt ${file} --device "cuda:1" --root ${root} \
									  --SenMap_dir ${sens_dir}  

	done
done
