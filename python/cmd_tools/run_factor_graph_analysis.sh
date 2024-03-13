#!/bin/bash


data_folder="/data/jagat/processed/"
eval_set=(
	MH_01_easy
   	MH_02_easy
   	MH_03_medium
   	MH_04_difficult
    MH_05_difficult
    V1_01_easy
    V1_02_medium
    V1_03_difficult
    V2_01_easy
    V2_02_medium
    V2_03_difficult
)

# mono
for dir_name in ${eval_set[@]}; do
    python ../factor_graph_insights/app.py -d $data_folder/$dir_name/factor_graphs/mono/lba/ -s 0 -e 35\
	 --loglevel info --near_depth_threshold 0.25 --far_depth_threshold 2 --plot
done

# # stereo
# for dir_name in ${eval_set[@]}; do
#     python ../factor_graph_insights/app.py -d $data_folder/$dir_name/factor_graphs/stereo/lba/ -s 0 -e 30 --loglevel info --near_depth_threshold 0.25 --far_depth_threshold 2
# done
