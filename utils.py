import os 
import config
import numpy as np
import pandas as pd

"""Auxiliary functions"""

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def format_save_results(lamdas_per_run):
    if config.mode == 'control':
        format_save_results_control(lamdas_per_run)
    else:
        format_save_results_test(lamdas_per_run)

def format_save_results_test(lamdas_per_run):
    lambdas = []
    for run, lambdas_dict in enumerate(lamdas_per_run):
        for lam in lambdas_dict.keys():
            harm_CF = lambdas_dict[lam]['hat_H']
            lambdas.append((run, 'CF', lam, harm_CF))
            harm_bound_cI = harm_CF + lambdas_dict[lam]['hat_G']
            lambdas.append((run, 'cI', lam, harm_bound_cI))
    
    lamdas_df = pd.DataFrame(lambdas, columns=["run", "assumption", "lamda", "empirical harm"])
    lamdas_df.set_index("run", inplace=True)

    path = f"{config.results_path}/{config.model_name}/noise{config.noise_level}/"
    if config.HUMAN_DATASET == "PS":
        path+=f"{config.HUMAN_DATASET}/"
    create_path(path)

    lamdas_df.to_csv(f"{path}/test_split{config.calibration_split}.csv", mode='w', header=False)

def format_save_results_control(lamdas_per_run):    
    lambdas = []
    for run, lambdas_dict in enumerate(lamdas_per_run):
        for assum in ['CF', 'cI']:
            for lam in lambdas_dict.keys():
                min_alpha = lambdas_dict[lam][assum]
                for alpha in np.arange(min_alpha, 1+config.alpha_step, config.alpha_step):
                    lambdas.append((run, assum, lam, np.round(alpha, decimals=2)))


    lamdas_df = pd.DataFrame(lambdas, columns=["run", "assumption", "lamda", "control level"])
    lamdas_df.set_index("run", inplace=True)

    path = f"{config.results_path}/{config.model_name}/noise{config.noise_level}/"
    if config.HUMAN_DATASET == "PS":
        path+=f"{config.HUMAN_DATASET}/"
    create_path(path)

    lamdas_df.to_csv(f"{path}/control_split{config.calibration_split}.csv", mode='w', header=False)

def format_save_metrics(lamdas_per_run):
    lamdas_metrics = []
    for run, lambdas_dict in enumerate(lamdas_per_run):
        for lam, metrics_dict in lambdas_dict.items():
            run_lam_metrics = (run, lam, metrics_dict['successes'])
            lamdas_metrics.append(run_lam_metrics)

    columns = ["run", "lambda", "successes"]
    
    lamdas_metrics_df = pd.DataFrame(lamdas_metrics, columns=columns)
    lamdas_metrics_df.set_index("run", inplace=True)

    path = f"{config.results_path}/{config.model_name}/noise{config.noise_level}/"
    if config.HUMAN_DATASET == "PS":
        path+=f"{config.HUMAN_DATASET}/"
    
    create_path(path)

    if config.HUMAN_DATASET == "PS":
        lamdas_metrics_df.to_csv(f"{path}/metrics_split{config.calibration_split}.csv", mode='w', header=False)
    elif config.MNL_PS:
        lamdas_metrics_df.to_csv(f"{path}/metrics_stratas2_PS_split{config.calibration_split}.csv", mode='w', header=False)
    else:
        lamdas_metrics_df.to_csv(f"{path}/metrics_stratas2_split{config.calibration_split}.csv", mode='w', header=False)

    