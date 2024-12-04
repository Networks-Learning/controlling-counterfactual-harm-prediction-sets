import config
import pandas as pd
from counterfactual_harm import *
import numpy as np 
from tqdm import tqdm
from utils import format_save_metrics, saps
import json


"""Estimates the average accuracy (either real or predicted using a mixture of MNLs) for each random sampling 
   of the test and calibration set for each lambda value given a fixed classifier and noise level"""

# Load dataset of images and true labels
x_y = pd.read_csv(f"{config.DATA_PATH}/imagenet16H.csv", dtype=str, index_col='image_name')

# Human predictions from prediction sets
if config.HUMAN_DATASET=="PS":
    human_pred_sets = pd.read_csv(f"{config.DATA_PATH}/study_pred_sets/rewards_strict.csv", index_col='image_name') 
elif config.MNL_PS:
    human_pred_sets = pd.read_csv(f"{config.DATA_PATH}/users/noise{config.noise_level}/model{config.model_name}/success_prob_stratas2_PS.csv", index_col='image_name') 
else:
    if config.score_type == 'vanilla':
        human_pred_sets = pd.read_csv(f"{config.DATA_PATH}/users/noise{config.noise_level}/model{config.model_name}/success_prob_stratas2.csv", index_col='image_name') 
    else:
        human_pred_sets = {}
        for weight in config.SAPS_WEIGHTS:
            human_pred_sets[weight] = pd.read_csv(f"{config.DATA_PATH}/users/noise{config.noise_level}/model{config.model_name}/success_prob_stratas2_saps_{weight}.csv", index_col='image_name') 

# Model predictions
if config.score_type == 'vanilla':
    model = pd.read_csv(f"{config.DATA_PATH}/models/noise{config.noise_level}/{config.model_name}.csv", index_col='image_name')
    model.drop(columns=['correct'], inplace=True)
else:
    model = {}
    for weight in config.SAPS_WEIGHTS:
        model[weight] = pd.read_csv(f"{config.DATA_PATH}/models/noise{config.noise_level}/{config.model_name}_saps_{weight}.csv", index_col='image_name')
        model[weight].drop(columns=['correct'], inplace=True)

# h function value on all human predictions (needed to get the stored model score on the true label for each image)
if config.score_type == 'vanilla':
    h = pd.read_csv(f"{config.DATA_PATH}/h_vanilla/noise{config.noise_level}/{config.model_name}{'_pred_set' if config.HUMAN_DATASET=='PS' else ''}.csv", index_col='image_name')
    model_score_true_label_df = h[['model_score_true_label', 'model_score_max']].drop_duplicates()
else:
    h = pd.read_csv(f"{config.DATA_PATH}/h_SAPS/noise{config.noise_level}/{config.model_name}{'_pred_set' if config.HUMAN_DATASET=='PS' else ''}.csv", index_col='image_name')
    model_score_true_label_df = h[['true_label_rank', 'model_score_max',]].drop_duplicates()

# Calibration set size
cal_size = int(len(x_y) * config.calibration_split)

# Create different generators for each run
generators = config.numpy_rng.spawn(config.N_RUNS)
tune_generators = config.tune_rng.spawn(config.N_RUNS)
    
# Keep lamdas per run
lamdas_per_run = []

for run in tqdm(range(config.N_RUNS)):
    
    # Get random generators for the current run
    config.numpy_rng = generators[run]
    config.tune_rng = tune_generators[run]

    # Sample calibration set
    calibration_set = x_y.sample(n=cal_size, random_state=config.numpy_rng)

    if config.score_type == 'SAPS':
        tuning_size = int(len(x_y) * config.tuning_split)
        assert config.tuning_split + config.calibration_split < 1.

        tune_set = x_y.drop(calibration_set.index).sample(n=tuning_size, random_state=config.tune_rng)
        # Load tuned saps weights per run 
        weight_per_run_path = f"{config.results_path}/{config.model_name}/noise{config.noise_level}/saps/saps_weights_tune{config.tuning_split}_run{run}.json" 
        assert  os.path.exists(weight_per_run_path), "SAPS weights do not exist"
        with open(weight_per_run_path, 'rt') as f:
            saps_weights = json.load(f)   
             
    # Sample the test set 
    test_set = x_y.drop(calibration_set.index)
    if config.score_type == 'SAPS':
        test_set = test_set.drop(tune_set.index)

    if config.score_type == 'vanilla':
        model_test = model.loc[test_set.index.values]
        model_score_true_label_df_test = model_score_true_label_df.loc[test_set.index.values]

        human_pred_sets_test_set = human_pred_sets.loc[test_set.index.values]
        human_pred_sets_test_set = human_pred_sets_test_set.reset_index().set_index(['image_name', 'set'])
    else:
        human_pred_sets_test_set = {}
        model_test = {}
        for w in human_pred_sets.keys():
            human_pred_sets_test_set[w] = human_pred_sets[w].loc[test_set.index.values]
            human_pred_sets_test_set[w] = human_pred_sets_test_set[w].reset_index().set_index(['image_name', 'set'])
            
            model_test[w] = model[w].loc[test_set.index.values]
            
        model_score_true_label_df_test = model_score_true_label_df.loc[test_set.index.values]
        
    # Compute the empirical metrics test_set
    lambdas_dict = { np.around(lam, decimals=config.lamda_dec) : {} for lam in np.arange(config.lambda_min, config.lambda_max + config.lambda_step, config.lambda_step)}
    for lamda in tqdm(lambdas_dict.keys()):
        
        # Prediction set sizes per data sample 
        if config.score_type == 'vanilla':
            images_set_sizes_with_empty = model_test.where(model_test >= 1 - lamda).count(axis=1).to_frame(name='set')    
        else:
            lam_weight = saps_weights[str(lamda)]
            images_set_sizes_with_empty = model_test[lam_weight].where(model_test[lam_weight] <= lamda).count(axis=1).to_frame(name='set')    
            
        # Replace empty sets with singletons
        images_set_sizes = images_set_sizes_with_empty.where(images_set_sizes_with_empty['set'] > 0, 1)
        set_sizes = images_set_sizes['set'].to_list()
        lambdas_dict[lamda]['set_sizes'] = set_sizes

        # Empirical coverage data sample 
        if config.score_type == 'vanilla':
            is_valid_set = ((model_score_true_label_df_test['model_score_true_label']>= (1 - lamda)) | (model_score_true_label_df_test['model_score_true_label'] == model_score_true_label_df_test['model_score_max'])).to_list()
        else:
            is_valid_set = (( 
                saps( 
                    lam_weight,
                    model_score_true_label_df_test['true_label_rank'],
                    model_score_true_label_df_test['model_score_max'],
                    config.saps_rng) 
                <= lamda ) | (model_score_true_label_df_test['true_label_rank'] == 1)).tolist()
        lambdas_dict[lamda]['coverage'] = is_valid_set

        # Empirical accuracy
        lambdas_dict[lamda]['successes'] = []
        images_set_sizes = images_set_sizes.reset_index().set_index(['image_name', 'set'])
        if config.score_type == 'vanilla':
            rewards =  human_pred_sets_test_set.loc[images_set_sizes.index]['reward']
        else:
            rewards =  human_pred_sets_test_set[lam_weight].loc[images_set_sizes.index]['reward']
       
        if config.HUMAN_DATASET=="PS":
            n_successes = rewards.sum()
            n_pred = rewards.count()
            lambdas_dict[lamda]['successes'] = [n_successes, n_pred]
        else:
            rewards_list = rewards.to_list()
            lambdas_dict[lamda]['successes'] = rewards_list

    lamdas_per_run.append(lambdas_dict)

format_save_metrics(lamdas_per_run)