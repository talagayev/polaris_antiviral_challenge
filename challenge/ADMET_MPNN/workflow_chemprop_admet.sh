#!/bin/bash

# Activate the chemprop Conda environment
#source activate chemprop

# Change the working directory
# cd ~/cyp_related_coding/chemprop/chemprop

# Define file paths
train_path="/mdspace/polaris_ml_challenge/chemprop_model_sijie/train_test_split/train_admet.csv"
test_path="/mdspace/polaris_ml_challenge/chemprop_model_sijie/train_test_split/test_admet.csv"
full_path="/mdspace/polaris_ml_challenge/chemprop_model_sijie/train_test_split/full_admet.csv"
save_dir="/mdspace/polaris_ml_challenge/chemprop_model_sijie/models/admet_chemprop_checkpoints"
conf_dir="/mdspace/polaris_ml_challenge/chemprop_model_sijie/models/admet_chemprop_configs"
#results_dir="./results/HLM_chemprop_results"

# hyperparameter optimization
chemprop_hyperopt --data_path "${train_path}" --dataset_type regression --num_iters 50 --config_save_path "${conf_dir}/opt_hyper_config.json"

# train the model with best hyperparameters
chemprop_train --data_path "${train_path}" --dataset_type regression --config_path "${conf_dir}/opt_hyper_config.json" \
                --save_dir "${save_dir}" --ensemble_size 10

# test the model with test set
chemprop_predict --test_path "${test_path}" \
   --checkpoint_dir "${save_dir}" \
   --preds_path "${results_dir}/test_performance.csv"

# train the model with all data
chemprop_train --data_path "${full_path}" --dataset_type regression --config_path "${conf_dir}/opt_hyper_config.json" \
                --save_dir "${save_dir}/full_model/" --ensemble_size 10

# predict with predset
chemprop_predict --test_path /mdspace/polaris_ml_challenge/chemprop_model_sijie/predset_admet_chemprop.csv \
--checkpoint_dir "${save_dir}/full_model/"  \
--preds_path "${results_dir}/predset_admet_chemprop_predicted.csv"
