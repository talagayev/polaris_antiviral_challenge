# Bayesian Neural Network (BNN) Models for Antiviral Potency Prediction

This folder contains training code, evaluation/prediction notebooks, config files, and pretrained models for Bayesian Neural Network (BNN) approaches used in the manuscript 'Fingerprint-Based Machine Learning for SARS-CoV-2 and MERS-CoV Mpro Inhibition: Highlighting the Potential of Bayesian Neural Networks'.

## Structure

- `MERS/`: BNN models and evaluation scripts for MERS-CoV potency prediction.
  - `train_BNN_MERS.py`: Training script for the MERS BNN model.
  - `BNN_eval_MERS.ipynb`: Jupyter notebook for evaluating the trained MERS BNN model.
  - `MERS_potency_BNN_complete_config_final.json`: Configuration file for the MERS BNN model.
  - `MERS_potency_BNN_model_final_all_data.pth`: Pretrained MERS BNN model weights.
- `SARS/`: BNN models and evaluation scripts for SARS-CoV potency prediction.
  - `train_BNN_SARS.py`: Training script for the SARS BNN model.
  - `BNN_eval_SARS.ipynb`: Jupyter notebook for evaluating the trained SARS BNN model.
  - `SARS_potency_BNN_complete_config_final.json`: Configuration file for the SARS BNN model.
  - `SARS_potency_BNN_model_final_all_data.pth`: Pretrained SARS BNN model weights.

## Usage

1. **Training**:  
   Use the `train_BNN_MERS.py` or `train_BNN_SARS.py` scripts to train a BNN model on your data.

2. **Evaluation**:  
   Use the `BNN_eval_MERS.ipynb` or `BNN_eval_SARS.ipynb` notebooks to evaluate the trained models and reproduce results. (As far as possible, given the random nature of BNN weight sampling.)

## Requirements

- PyTorch
- torchbnn
- RDKit
- tqdm
- optuna
- scikit-learn
- matplotlib
- pandas
- scipy
- numpy

