preliminary results
- BNN model trained with:
  - Input = Morgan finger print radius 2 calculated with rdkit
  - bayesian neural network with three layers (prior_mu=0, prior_sigma=0.1)
    - 1 layer: 2048 in-features, 50 out-features, 
    - 2 layer: 50 in-featues, 25 out-features
    - 3 layer: 25 in-featues, 1 out-features
- input morgan FP calculated based on SMILES
- trained on the "SARS-CoV-2-MPro_fluorescence-dose-response_weizmann: pIC50 (log10M)" values
- final prediction run with 100 forward passes that were then averaged for the final result
