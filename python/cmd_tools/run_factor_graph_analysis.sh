#!/bin/bash

if [ $# -ne 4 ]; then
    echo " Usage: ./run_factor_graph_analysis.sh <start_file> <end_file> <far_depth_mono> <far_depth_stereo>"
    echo " Default values for far_depth for both mono and stereo is 2"
    exit 0
fi

start=$1
end=$2
far_depth_mono=$3
far_depth_stereo=$4

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
    echo "Covariance calculation for the ${dir_name} - mono data"
    python ../factor_graph_insights/app.py -d $data_folder/$dir_name/factor_graphs/mono/lba/ -s $start -e $end\
	 --loglevel info --near_depth_threshold 0.25 --far_depth_threshold $far_depth_mono --plot &
done

# stereo
for dir_name in ${eval_set[@]}; do
    echo "Covariance calculation for the ${dir_name} - stereo data"
    python ../factor_graph_insights/app.py -d $data_folder/$dir_name/factor_graphs/stereo/lba/ -s $start -e $end\
	 --loglevel info --near_depth_threshold 0.25 --far_depth_threshold $far_depth_stereo --plot &
done

wait
echo " All processes finish "

exit 0