# Polaris_antiviral_challenge
This Github repository is part of the [antiviral competition](https://openadmet.org/blog/asap-blind-challenge/) hosted on [Polaris](https://polarishub.io/competitions) consisting of three sub challenges.

# Potency prediction

For the potency predictions two separated approaches were used consisting of an standard XGBoost ML approach, used for MERS-COV and
a bayesian neural network for the SARS-COV2

## SARS-COV2

- BNN model trained with:
  - Input = Morgan finger print radius 2 calculated with rdkit
  - bayesian neural network with three layers (prior_mu=0, prior_sigma=0.1)
    - 1 layer: 2048 in-features, 50 out-features, 
    - 2 layer: 50 in-featues, 25 out-features
    - 3 layer: 25 in-featues, 1 out-features
- input morgan FP calculated based on SMILES
- trained on the "SARS-CoV-2-MPro_fluorescence-dose-response_weizmann: pIC50 (log10M)" values
- final prediction run with 100 forward passes that were then averaged for the final result

The notebook and files used for this prediction can be found at the:
[BNN SARS-COV2 folder](https://github.com/talagayev/polaris_antiviral_challenge/tree/main/SARS_COV2_potency_BNN)

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
