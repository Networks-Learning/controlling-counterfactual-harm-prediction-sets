import config
import pandas as pd
from counterfactual_harm import *
from tqdm import tqdm
from utils import format_save_results

"""For a given model and noise level sample randomly test and calibration set per 
    iteration and compute for each lambda value all controlling levels (alpha)
    for the h function (Thm 1) and the g function (Thm 2).
    """

# Load dataset of images and true labels
x_y = pd.read_csv(f"{config.DATA_PATH}/imagenet16H.csv", dtype=str, index_col='image_name')

# Human predictions
file_post_fix = "_pred_set" if config.HUMAN_DATASET=="PS" else ""
human = pd.read_csv(f"{config.DATA_PATH}/users/noise{config.noise_level}/success{file_post_fix}.csv", index_col='image_name') 

# Model predictions
model = pd.read_csv(f"{config.DATA_PATH}/models/noise{config.noise_level}/{config.model_name}.csv", index_col='image_name')

# Harm module
cf_harm = CounterfactualHarm(model, human, x_y)

# Calibration set size
cal_size = int(len(x_y) * config.calibration_split)

# Create different generators for each run
generators = config.numpy_rng.spawn(config.N_RUNS)
    
# Keep controlling lamdas per run
lamdas_per_run = []

for run in tqdm(range(config.N_RUNS)):
    
    # Get random generator for the current run
    config.numpy_rng = generators[run]

    # Sample calibration set
    calibration_set = x_y.sample(n=cal_size, random_state=config.numpy_rng)
    
    if config.mode == 'control':
        # Set up calibration set for harm control
        cf_harm.set_data_set(calibration_set)

        # Dict of lambda values with the min controlling levels for g and h 
        lambdas_dict = cf_harm.control()
    else:    
        # Sample the test set 
        test_set = x_y.drop(calibration_set.index)

        # Set up the test set
        cf_harm.set_data_set(test_set)
        
        # Compute the empirical harm on the test set
        lambdas_dict = cf_harm.compute()

    lamdas_per_run.append(lambdas_dict)

# Save for lambda all controlling levels for h and g
format_save_results(lamdas_per_run)