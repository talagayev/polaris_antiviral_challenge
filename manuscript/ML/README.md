# Traditional Machine Learning (ML) Models for Antiviral Potency Prediction

This folder contains training code, evaluation/prediction notebooks, and pretrained models for the traditional machine learning (ML) approaches used in the manuscript 'Fingerprint-Based Machine Learning for SARS-CoV-2 and MERS-CoV Mpro Inhibition: Highlighting the Potential of Bayesian Neural Networks'.

## Structure

- `MERS/`: ML models and evaluation scripts for MERS-CoV potency prediction.
  - `MERS_potency_calculation_XG.ipynb`: Jupyter notebook for evaluating the trained MERS XG model.
  - `MERS_potency_calculation_RF.ipynb`: Jupyter notebook for evaluating the trained MERS RF model.
  - `class_models/`: Folder with the pretrained RF and XG models for MERS-CoV potency prediction
    - `XG_models_clust_mers_test2`: Folder with the pretrained XG model for MERS-CoV potency prediction
    - `RF_models_clust_mers_test`: Folder with the pretrained RF model for MERS-CoV potency prediction
- `SARS/`:  ML models and evaluation scripts for SARS-CoV potency prediction.
  - `SARS_potency_calculation_XG.ipynb`: Jupyter notebook for evaluating the trained SARS XG model.
  - `SARS_potency_calculation_RF.ipynb`: Jupyter notebook for evaluating the trained SARS RF model.
  - `class_models/`: Folder with the pretrained RF and XG models for SARS-CoV-2 potency prediction
    - `XG_models_clust_sars_test`: Folder with the pretrained XG model for SARS-CoV-2 potency prediction
    - `RF_models_clust_sars_test`: Folder with the pretrained RF model for SARS-CoV-2 potency prediction
- `XGBoost_RF_Test_Set_Evaluation.ipynb`:  Evaluation script for the predictions.
- `mers_predicted_true_xg.csv`:  XG model predictions for MERS-CoV.
- `mers_predicted_true_rf.csv`:  RF model predictions for MERS-CoV.
- `sars_predicted_true.csv`:  XG model predictions for SARS-CoV-2.
- `sars_predicted_true_rf.csv`:  RF model predictions for SARS-CoV-2.

## Usage

1. **Training**:  
   Run the `MERS_potency_calculation_XG.ipynb` and `MERS_potency_calculation_RF.ipynb` notebooks for the MERS predictions and the `SARS_potency_calculation_XG.ipynb` and `SARS_potency_calculation_RF.ipynb`  notebooks for the SARS predictions.

2. **Evaluation**:  
   Use the `XGBoost_RF_Test_Set_Evaluation.ipynb` notebooks to evaluate the trained models.

## Requirements

- QSPRpred
- RDKit
- optuna
- scikit-learn
- matplotlib
- pandas
- scipy
- numpy

