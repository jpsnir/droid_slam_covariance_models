#!/bin/bash

data_folder="/data/jagat/processed"

relative_data_paths=(
    "MH_01_easy/factor_graphs/mono/lba/fg_0029_id_0206.pkl"
    "MH_02_easy/factor_graphs/mono/lba/fg_0048_id_0404.pkl"
    "MH_04_difficult/factor_graphs/mono/lba/fg_0038_id_0208.pkl"
    "MH_05_difficult/factor_graphs/mono/lba/fg_0025_id_0152.pkl"
    "V1_01_easy/factor_graphs/mono/lba/fg_0027_id_0218.pkl"
    "V1_03_difficult/factor_graphs/mono/lba/fg_0031_id_0112.pkl"
    "V2_03_difficult/factor_graphs/mono/lba/fg_0055_id_0172.pkl"
)

for rel_file_path in ${relative_data_paths[@]}; do
    echo " Processing file ${rel_file_path}"
    python ../factor_graph_insights/app_2.py -f $data_folder/$rel_file_path --log_to_file --far_depth_threshold 2 &
done
wait
echo "Processing complete"