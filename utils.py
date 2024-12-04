import os 
import config
import numpy as np
import pandas as pd
from copy import deepcopy

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
            run_lam_metrics = (run, lam)
            for metric in ['set_sizes', 'coverage', 'successes']:
                run_lam_metrics+=(metrics_dict[metric],)
            lamdas_metrics.append(run_lam_metrics)

    columns = ["run", "lambda"]
    for metric in ['set_sizes', 'coverage', 'successes']:
        columns.append(metric)
    
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


def saps(weight, label_rank, max_score, rng):
    u = rng.uniform(size=label_rank.shape)
    tmp = deepcopy(max_score)
    scores = tmp.where(label_rank == 1, weight * (label_rank - 2 + u) + max_score)
    scores = scores.where(label_rank > 1, u * max_score)
    return scores

def saps_batch(weight, probs, label_ranks, max_scores):
    max_scores_matrix = deepcopy(label_ranks)
    for col in max_scores_matrix.columns:
        max_scores_matrix[col] = max_scores
    u = np.random.uniform(size=probs.shape)
    scores = probs.where(label_ranks == 1, other=weight * (label_ranks - 2 + u) + max_scores_matrix)
    scores = scores.where(label_ranks > 1, other=u*max_scores_matrix)
    return scores

def call_saps(weight, probs, max_scores):
    label_ranks = deepcopy(probs)
    for idx in probs.index:
        label_ranks.loc[idx] = config.N_LABELS - probs.loc[idx].argsort().argsort()
    scores = saps_batch(weight, probs, label_ranks, max_scores)
    return scores