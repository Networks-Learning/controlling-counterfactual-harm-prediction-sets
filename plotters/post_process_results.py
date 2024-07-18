import config
import pandas as pd
import numpy as np
from copy import deepcopy
import os

"""Post process saved results from experiments required for plotting"""

def read_results(path, keys):
    print(path)
    df = pd.read_csv(path, names=keys, header=None)
    return df

METRICS_MAPPER = {
    'harm': lambda x: np.array(list(map(int, x[1:-1].split(',')))),
    'accuracy': lambda x: np.array(list(map(int, x[1:-1].split(',')))), 
    'accuracy_mnl': lambda x: np.array(list(map(float, x[1:-1].split(',')))),
    }
METRICS_GROUPBY_KEYS = {
    'harm': ['assumption', 'lambda'],
    'metrics': ['lambda'],
    'metric_2_mnl': ['lambda'],
    'metric_2_PS_mnl': ['lambda'],
}
METRICS_FILE_KEY = {
    'harm': 'test',
    'metrics': 'metrics',
    'metric_2': 'metrics_stratas2', 
    'metric_2_PS': 'metrics_stratas2_PS', 
    }
METRICS_KEYS = {
    'harm': ['run', 'assumption', 'lambda', 'harm'],
    'metrics': ['run', 'lambda', 'accuracy'],
    'metric_2_mnl': ['run', 'lambda',  'accuracy_mnl'],
    'metric_2_PS_mnl': ['run', 'lambda',  'accuracy_mnl']
}
METRICS_MEAN = {
    'harm': lambda x: x[0] / x[1] if len(x)==2 else x[2]/x[3] + x[0]/x[1],#(x[0]+x[2]) / (x[1]),
    'accuracy': lambda x: x[0] / x[1], 
    'accuracy_mnl': lambda x: np.mean(x), 
}
METRICS_AGG = {
    'harm': 'sum',
    'accuracy': 'sum', 
    'accuracy_mnl': lambda x: np.concatenate(x.values), 
}
def se_fn(x):
    if len(x) == 2:
        harms = [1.]*x[0] + (x[1]-x[0])*[0.]
        return np.std(harms) / np.sqrt(x[1])
    else:
        harm_bound = [1.]*(x[0]+x[2]) + (x[1]-x[0] - x[2])*[0.]
        return np.std(harm_bound) / np.sqrt(x[1])
METRICS_SE = {
    'harm': se_fn,
    'accuracy': se_fn, 
    'accuracy_mnl': lambda x: np.std(x) / np.sqrt(len(x))
}
def compute_means_se(metric_name, base_path, file_postfix, pred_sets):
    """Computes mean and standard error of several metrics (harm/accuracy) per lambda value
    across iterations"""
    metric_path = f"{base_path}/{METRICS_FILE_KEY[metric_name]}_{file_postfix}"
    if not pred_sets and 'harm' not in metric_name:
        metric_name+='_mnl'
    print(metric_path)
    if 'harm' in metric_name:
        metric_name = 'harm'

    df = read_results(path=metric_path, keys=METRICS_KEYS[metric_name]).drop(columns=['run'])
   
    # TODO: make this a for loop 
    if metric_name == 'harm':
        metric_key = 'harm'
        map_dict = {'harm': METRICS_MAPPER['harm'] }
        agg_dict = {'harm': METRICS_AGG['harm'] }
        mean_dict = {'harm': METRICS_MEAN['harm'] }
        se_dict = {'harm': METRICS_SE['harm'] }
    else:
        metric_key = ['accuracy'] if pred_sets else  ['accuracy_mnl']
        map_dict = {metric_n: METRICS_MAPPER[metric_n]for metric_n in metric_key}
        agg_dict = {metric_n: METRICS_AGG[metric_n]for metric_n in metric_key}
        mean_dict = {metric_n: METRICS_MEAN[metric_n]for metric_n in metric_key}
        se_dict = {metric_n: METRICS_SE[metric_n]for metric_n in metric_key}
    
    df[metric_key] = df.apply(map_dict)

    df = df.groupby(METRICS_GROUPBY_KEYS[metric_name]).aggregate(agg_dict)
   
    if metric_key=='harm':
        df[f"{metric_key}_mean"] = df.apply(mean_dict)
        df[f"{metric_key}_se"] = df.apply(se_dict)
        df.drop(columns=metric_key, inplace=True)
    else:
        df[[f"{metric_n}_mean" for metric_n in metric_key]] = df.apply(mean_dict)
        df[[f"{metric_n}_se" for metric_n in metric_key]] = df.apply(se_dict)
        for metric_n in metric_key:
            df.drop(columns=metric_n, inplace=True)
        if not pred_sets:
            df.rename(
                {f"accuracy_mnl_{m}":f"accuracy_{m}" for m in ["mean","se"]},
                axis=1, 
                inplace=True
            )

    return df

def load_or_compute_metric(metric, pred_sets=True, model_name=config.model_name, noise_level=config.noise_level):
    """Computes or loads saved mean and se values of metrics per lambda value across iterations
    for fixed model and noise level"""
    base_path = f"{config.ROOT_DIR}/{config.results_path}/{model_name}/noise{noise_level}/"
    post_fix = f"{'/PS' if pred_sets else '' }"
    base_path+=post_fix
    file_postfix = f"split{config.calibration_split}.csv"

    metrics_mean_se_path = f"{base_path}/{metric}_mean_se_{file_postfix}"
    if not os.path.exists(metrics_mean_se_path):
        df_means_se = compute_means_se(metric, base_path, file_postfix, pred_sets)
        df_means_se.reset_index(inplace=True)
        df_means_se.to_csv(metrics_mean_se_path, header=True)
    else:
        df_means_se = pd.read_csv(metrics_mean_se_path, header=0, index_col=0)
        
    return df_means_se

def metric_vs_harm(
        pred_sets=True, 
        metric='metric_2', 
        model_name=config.model_name, 
        noise_level=config.noise_level, 
        alpha=0.01
        ):
    """Returns results to be plotted"""
    m_mean_se_dict = {}
    if model_name == 'Real' or model_name == 'Predicted':
        model_name = 'vgg19'
    for m in ['harm', metric]:
        m_mean_se_dict[m] = load_or_compute_metric(m, pred_sets=pred_sets, model_name=model_name, noise_level=noise_level)

    m_mean_se_dict['harm'].set_index(['assumption', 'lambda'], inplace=True)
    if 'harm' not in metric:
        m_mean_se_dict[metric].set_index('lambda', inplace=True)
        harm_metrics = m_mean_se_dict['harm'].join(m_mean_se_dict[metric], how='left')
    elif metric!='harm':
        m_mean_se_dict[metric].set_index([
                                  'assumption', 
                                  'lambda'], 
                                  inplace=True)
        harm_metrics = m_mean_se_dict[metric]
    else:
        harm_metrics = m_mean_se_dict['harm']
    

    lambda_idx_cnt = find_lambda_idx(pred_sets=pred_sets, model_name=model_name, alpha=alpha)


    harm_metrics_lamda_freq = harm_metrics.join(lambda_idx_cnt, how='left', on=['assumption', 'lambda']).fillna(0)
    return harm_metrics_lamda_freq


def find_lambda_idx(pred_sets=True, model_name=config.model_name, alpha=config.alpha):
    """Computes the intesity of the coloring of 
    each lambda value under each monotonicity assumption"""
    base_path = f"{config.ROOT_DIR}/{config.results_path}/{model_name}/noise{config.noise_level}"
    post_fix = f"{'/PS' if pred_sets else '' }"
    base_path+=post_fix

    file_postfix = f"split{config.calibration_split}.csv"

    lamdas_file_path = f"{base_path}/control_{file_postfix}"

    lamdas = read_results(lamdas_file_path, ['run', 'assumption', 'lambda','alpha'])
    # alpha here works as flag to know when asking results on CF monotonicity
    # TODO: pass assum as parameter 
    if alpha<=0.1:
        lamdas_for_alpha = lamdas[lamdas['alpha']==alpha].drop(columns=['alpha'])

        # Coloring for counterfactual monotonicity
        control_across_runs = lamdas_for_alpha.groupby(['assumption', 'lambda']).count()
        return control_across_runs
    
    # Coloring for interventional monotonicity
    a_prime = 0.
    max_lam_per_iter = 0
    max_ctr = 0
    a_max = 0
    while a_prime <= alpha:
        
        if alpha-a_prime< 1./121:
            a_prime+=0.01
            continue
        
        lamdas_for_alpha_CF = lamdas[
            (lamdas['alpha']==np.round(a_prime, decimals=2))&(lamdas['assumption'] == 'CF')].set_index(['lambda', 'run'])
        
        if lamdas_for_alpha_CF.empty:
            a_prime+=0.01
            continue
        lamdas_for_alpha_cI = lamdas[ 
            (((lamdas['alpha']==np.round((alpha-a_prime), decimals=2))&(lamdas['assumption'] == 'cI')))].set_index(['lambda', 'run'])
        if lamdas_for_alpha_cI.empty:
            a_prime+=0.01
            continue
       
        inters_idx = lamdas_for_alpha_CF.index.unique().intersection(lamdas_for_alpha_cI.index.unique())
        if inters_idx.empty:
            a_prime+=0.01
            continue
        lamdas_for_alpha = lamdas_for_alpha_cI.loc[inters_idx]

        ctr_runs = lamdas_for_alpha.reset_index().groupby('lambda').count()
        lam_per_iter = lamdas_for_alpha.reset_index().groupby('lambda').count()['run'].mean(axis=0) 
        if lam_per_iter > max_lam_per_iter:
            max_lam_per_iter = deepcopy(lam_per_iter)
            max_ctr = deepcopy(ctr_runs)
            a_max = a_prime
 
    assum_lam_n_run = [('cI', lam, 0) for lam in lamdas['lambda'].unique()]
    control_across_runs = pd.DataFrame(assum_lam_n_run,
                                       columns=['assumption', 'lambda', 'run'], 
                                        ).set_index(['assumption','lambda'])
    # Picked alpha'
    print(a_max)
    for lam, row in max_ctr.iterrows():
        idx = pd.MultiIndex.from_tuples([('cI', lam)])
        control_across_runs.loc[idx, 'run'] = row['run']

    return control_across_runs
   
