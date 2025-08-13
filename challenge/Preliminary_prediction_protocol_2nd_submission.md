This file serves as a protocol for the second intermediate prediction for the Antiviral challenge.

The only change from the first indermediate submission consists in the attempt to apply ML for the prediction of the SARS-COV potency.

# Potency prediction

For the potency predictions two separated approaches were used consisting of an standard XGBoost ML approach, used for MERS-COV and
SARS-COV2

## SARS-COV2

- For this prediction [QSPRPRED](https://github.com/CDDLeiden/QSPRpred) was used
- The XGBoost model with a 80% training and 20% test split.
- Fingerprints: MorganFP(radius=3, nBits=1024), RDKitFP(maxPath=8, nBits=256)
- Hyperparameter optimization: Optuna Optimizer with the following search space:
- max_depth: [1, 20] 
- gamma: [0, 20]
- max_delta_step: [0, 20]
- min_child_weight: [1, 20]
- learning_rate: [0.001, 1]
- subsample: [0.001, 1]
- n_estimators: [10, 250]
- for the splitting Clustersplit was used

The notebook and files used for this prediction can be found at the:
[Intermediate prediction 2 folder](https://github.com/talagayev/polaris_antiviral_challenge/tree/main/Intermediate_submission_2)

## MERS-COV

- For this prediction [QSPRPRED](https://github.com/CDDLeiden/QSPRpred) was used
- The XGBoost model with a 80% training and 20% test split.
- Fingerprints: MorganFP(radius=3, nBits=1024), RDKitFP(maxPath=7, nBits=512)
- Hyperparameter optimization: Optuna Optimizer with the following search space:
- max_depth: [1, 20] 
- gamma: [0, 20]
- max_delta_step: [0, 20]
- min_child_weight: [1, 20]
- learning_rate: [0.001, 1]
- subsample: [0.001, 1]
- n_estimators: [10, 250]
- for the splitting Clustersplit was used

The notebook and files used for this prediction can be found at the:
[MERS_COV_potency_XG_ML folder](https://github.com/talagayev/polaris_antiviral_challenge/tree/main/MERS_COV_potency_XG_ML)

# ADMET Predictions

- An MPNN regression model trained using Chemprop version 1.7.0.
- Input = preprocessed smiles with ChEMBL structural pipeline Dropped duplicates based on molecule name in each subset and merged all admet subset in to one.
- 4:1 train test split
- Hyperparameter optimization 50 iteration
- Final model trained with all data available (train + test) with 10 ensemble

The notebook and files used for this prediction can be found at the:
[ADMET_MPNN Folder](https://github.com/talagayev/polaris_antiviral_challenge/tree/main/ADMET_MPNN)
