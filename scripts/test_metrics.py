import config
import pandas as pd
from counterfactual_harm import *
import numpy as np 
from tqdm import tqdm
from utils import format_save_metrics

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
    human_pred_sets = pd.read_csv(f"{config.DATA_PATH}/users/noise{config.noise_level}/model{config.model_name}/success_prob_stratas2.csv", index_col='image_name') 

# Model predictions
model = pd.read_csv(f"{config.DATA_PATH}/models/noise{config.noise_level}/{config.model_name}.csv", index_col='image_name')
model.drop(columns=['correct'], inplace=True)

# h function value on all human predictions (needed to get the stored model score on the true label for each image)
h = pd.read_csv(f"{config.DATA_PATH}/h/noise{config.noise_level}/{config.model_name}{'_pred_set' if config.HUMAN_DATASET=='PS' else ''}.csv", index_col='image_name')
model_score_true_label_df = h[['model_score_true_label', 'model_score_max']].drop_duplicates()

# Calibration set size
cal_size = int(len(x_y) * config.calibration_split)

# Create different generators for each run
generators = config.numpy_rng.spawn(config.N_RUNS)
    
# Keep lamdas per run
lamdas_per_run = []

for run in tqdm(range(config.N_RUNS)):
    
    # Get random generator for the current run
    config.numpy_rng = generators[run]

    # Sample calibration set
    calibration_set = x_y.sample(n=cal_size, random_state=config.numpy_rng)
    
    # Sample the test set 
    test_set = x_y.drop(calibration_set.index)

    model_test = model.loc[test_set.index.values]
    model_score_true_label_df_test = model_score_true_label_df.loc[test_set.index.values]

    human_pred_sets_test_set = human_pred_sets.loc[test_set.index.values]
    human_pred_sets_test_set = human_pred_sets_test_set.reset_index().set_index(['image_name', 'set'])

    # Compute the empirical metrics test_set
    lambdas_dict = {lam : {} for lam in np.arange(0, 1 + config.lambda_step, config.lambda_step)}
    for lamda in tqdm(lambdas_dict.keys()):
        
        # Prediction set sizes per data sample 
        images_set_sizes_with_empty = model_test.where(model_test >= 1 - lamda).count(axis=1).to_frame(name='set')    

        # Replace empty sets with singletons
        images_set_sizes = images_set_sizes_with_empty.where(images_set_sizes_with_empty['set'] > 0, 1)

        # Emprical accuracy
        lambdas_dict[lamda]['successes'] = []
        images_set_sizes = images_set_sizes.reset_index().set_index(['image_name', 'set'])
        rewards =  human_pred_sets_test_set.loc[images_set_sizes.index]['reward']
       
        if config.HUMAN_DATASET=="PS":
            n_successes = rewards.sum()
            n_pred = rewards.count()
            lambdas_dict[lamda]['successes'] = [n_successes, n_pred]
        else:
            rewards_list = rewards.to_list()
            lambdas_dict[lamda]['successes'] = rewards_list

    lamdas_per_run.append(lambdas_dict)

format_save_metrics(lamdas_per_run)

