#!/bin/bash -e

# DATASET           (trainsize, testsize, ts_length)
# SwedishLeaf :     (500, 625, 128)
# ECG200 :          (100, 100, 96)
# Plane :           (105, 105, 144)
# ECG5000 :         (500, 4500, 140)
# Epilepsy2 :       (80, 11420, 178)

# explainers=("shapley" "kernelshap" "leftist_shap" "leftist_lime" "lime")

explainers=("leftist_shap" "leftist_lime")

for exp in "${explainers[@]}"; do

    python saliencyserieslab/generate_weights.py --model ./models/mrseql_ECG200_1 --explainer ${exp} 
    python saliencyserieslab/generate_weights.py --model ./models/rocket_ECG200_1 --explainer ${exp} 
    # python saliencyserieslab/generate_weights.py --model ./models/weasel_ECG200_1 --explainer ${exp} 

    python saliencyserieslab/generate_weights.py --model ./models/mrseql_SwedishLeaf_1 --explainer ${exp} 
    python saliencyserieslab/generate_weights.py --model ./models/rocket_SwedishLeaf_1 --explainer ${exp} 
    # python saliencyserieslab/generate_weights.py --model ./models/weasel_SwedishLeaf_1 --explainer ${exp} 

    python saliencyserieslab/generate_weights.py --model ./models/mrseql_Plane_1 --explainer ${exp} 
    python saliencyserieslab/generate_weights.py --model ./models/rocket_Plane_1 --explainer ${exp} 
    # python saliencyserieslab/generate_weights.py --model ./models/weasel_Plane_1 --explainer ${exp} 

done

explainers=("shapley" "kernelshap" "lime")

for exp in "${explainers[@]}"; do

    python saliencyserieslab/generate_weights.py --model ./models/mrseql_SwedishLeaf_1 --explainer ${exp} 
    python saliencyserieslab/generate_weights.py --model ./models/rocket_SwedishLeaf_1 --explainer ${exp} 
    # python saliencyserieslab/generate_weights.py --model ./models/weasel_SwedishLeaf_1 --explainer ${exp} 

done