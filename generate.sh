#!/bin/bash -e

# DATASET           (trainsize, testsize, ts_length)
# SwedishLeaf :     (500, 625, 128)
# ECG200 :          (100, 100, 96)
# Plane :           (105, 105, 144)
# ECG5000 :         (500, 4500, 140)
# Epilepsy2 :       (80, 11420, 178)

# explainers=("shapley" "kernelshap" "leftist_shap" "leftist_lime" "lime")
# models=("mrseql", "rocket", "resnet")

explainers=("leftist_shap" "leftist_lime" "shapley" "kernelshap" "lime")

for exp in "${explainers[@]}"; do

    python saliencyserieslab/generate_weights.py --model ./models/mrseql_ECG200_1 --explainer ${exp} 
    python saliencyserieslab/generate_weights.py --model ./models/mrseql_SwedishLeaf_1 --explainer ${exp} 
    python saliencyserieslab/generate_weights.py --model ./models/mrseql_Plane_1 --explainer ${exp}

    python saliencyserieslab/generate_weights.py --model ./models/resnet_ECG200_1 --explainer ${exp} 
    python saliencyserieslab/generate_weights.py --model ./models/resnet_SwedishLeaf_1 --explainer ${exp} 
    python saliencyserieslab/generate_weights.py --model ./models/resnet_Plane_1 --explainer ${exp}

    python saliencyserieslab/generate_weights.py --model ./models/rocket_ECG200_1 --explainer ${exp} 
    python saliencyserieslab/generate_weights.py --model ./models/rocket_SwedishLeaf_1 --explainer ${exp} 
    python saliencyserieslab/generate_weights.py --model ./models/rocket_Plane_1 --explainer ${exp} 
done

python saliencyserieslab/generate_mrseql_weights.py --model ./models/mrseql_ECG200_1
python saliencyserieslab/generate_mrseql_weights.py --model ./models/mrseql_SwedishLeaf_1
python saliencyserieslab/generate_mrseql_weights.py --model ./models/mrseql_Plane_1
