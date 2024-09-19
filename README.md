# IMOE
Code for the paper "Enhancing Explanations of Graph Neural Network via Bridging Model-level and Instance-level Explainers"

## Overview
The project contains the following folders and files.
- datasets: All datasets are contained in this fold. 
- codes
	- GNNmodels: Code for the GNN model to be explained.
 	- ModelLevelExplainer: Code for the model-level Explainer.
	- InstanceLevel_Explainer.py: Code for instance-level Explainer.
	- load_dataset.py: Load datasets.
	- Configures.py: Parameter configuration of the GNN model to be explained.
	- metrics.py: Metrics of the evaluation.
	- plot_utils.py: Plot a diagram of a case for instance-level explanations.
- models
  - gnnModel: To facilitate the reproduction of the experimental results in the paper, we provide the trained GNNs to be explained in this fold.
  - instance_level_explainer: The instance-level explainers trained on the BA-4Motifs dataset.
  - model_level_explainer: A model-level explainer trained on the BA-4Motifs dataset.
- prototype: The prototypes learned are contained in this fold. 

## Prerequisites
- python >= 3.9
- torch >= 1.12.1+cu113
- torch-geometric >= 2.2.0

## To run
- Run train_GNNNets.py to train the GNNs to be explained. Change parameter **dataset** per demand.
- Run train_model_level_explainer.py to train the model-level explainer. Change **dataset** per demand.
- Run cluster_prototype.py to generate model-level candidates and cluster to prototype. Change **dataset** per demand.
- Run instance_level_explanation.py to train and test the instance-level explainer. Change **parameters** per demand.

