# export root="/home/h2amer/AhmedH.Salamah/ilsvrc2012"
export root="/home/h2amer/work/workspace/ML_TS/"
# export root="~/data/ImageNet/2012"

# tune lambda 0.1 0.5 1.0
export Lambda=0.1

export beta=1
export colorspace=0
export model=NoModel
# export sens=SenMap_Normalized
export sens=NoModel

export file=./Resize_Compress/SDQ_OptD/${model}/YUV/${model}_B%.2f_sens_${sens}_d_water_Y%.4f_d_water_C%.4f_Q_max_Y%d_Q_max_C%d.txt
export sens_dir=./SenMap_All/${sens}/${model}
echo ${file}
python3 SDQ_OptD_test_dataloader_4models.py --Model ${model} --J 4 --a 4 --b 4 \
							  -resize_compress --colorspace ${colorspace} \
							  --Qmax_Y 46 --Qmax_C 46 --DT_Y 100 --DT_C 100 \
							  --d_waterlevel_Y 0 --d_waterlevel_C 0  \
							  --Beta_S ${beta}  --Beta_W ${beta}  --Beta_X ${beta} --L ${Lambda} \
							  --output_txt ${file} --device "cpu" --root ${root} \
							  --SenMap_dir ${sens_dir}  --OptD True
