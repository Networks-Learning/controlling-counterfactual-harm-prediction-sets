import config 
import numpy as np
import os
import pandas as pd
import json
from utils import saps, saps_batch, create_path
from copy import deepcopy
"""Main module to control and evaluate counterfactual harm (bound)"""

class CounterfactualHarm:
    def __init__(self, model, human, data, score_type='vanilla') -> None:
        self.model = model
        self.human = human
        self.data = data
        self.emp_risks = {}
        self.score_type = score_type
        self.H = self.fn_per_set("h")
        self.G = self.fn_per_set("g")
        self.alphas = np.arange(0,1+config.alpha_step, config.alpha_step)

    def _empirical_risk(self, lamda):
        """Computes hat H(\lambda)"""
        # Set is valid or corner case of empty set
        if self.score_type == 'vanilla':
            is_valid_set = (self.H_data_set['model_score_true_label'] >= (1 - lamda)) | (self.H_data_set['model_score_max'] == self.H_data_set['model_score_true_label'])
        else:
            lamda_k = str(np.around(lamda, decimals=config.lamda_dec))
            assert lamda_k in self.weights, self.weights.keys()
            is_valid_set = (saps(
                self.weights[lamda_k], 
                self.H_data_set['true_label_rank'],
                self.H_data_set['model_score_max'],
                config.saps_rng
                ) <= lamda) | \
            (self.H_data_set['true_label_rank'] == 1)
            
        if config.mode == 'control':
            emp_risk = self.H_data_set.where(is_valid_set, self.H_data_set['h_invalid_sets'], axis=0)['h_valid_sets'].mean()
            return emp_risk
        else:
            harm = self.H_data_set.where(is_valid_set, self.H_data_set['h_invalid_sets'], axis=0)['h_valid_sets']
            harm_sum = harm.sum()
            harm_count = harm.count()
            return harm_sum, harm_count
      

    def _empirical_benefit(self, lamda):
        """Computes hat G(\lambda)"""
        # Set is valid or corner case of empty set
        if self.score_type == 'vanilla':
            is_valid_set = (self.G_data_set['model_score_true_label'] >= (1 - lamda)) | (self.H_data_set['model_score_max'] == self.H_data_set['model_score_true_label'])
        else:
            lamda_k = str(np.around(lamda, decimals=config.lamda_dec))
            is_valid_set = (saps(
                self.weights[lamda_k], 
                self.G_data_set['true_label_rank'],
                self.G_data_set['model_score_max'],
                config.saps_rng
                ) <= lamda) | \
            (self.G_data_set['true_label_rank'] == 1)
           

        if config.mode == 'control':
            emp_ben = self.G_data_set.where(is_valid_set, self.G_data_set['g_invalid_sets'], axis=0)['g_valid_sets'].mean()
            return emp_ben
        else:
            benefit = self.G_data_set.where(is_valid_set, self.G_data_set['g_invalid_sets'], axis=0)['g_valid_sets']
            g_sum = benefit.sum()
            g_count = benefit.count()
            return (g_sum, g_count)

    
    def fn_per_set(self, fn_name):
        """Reads/computes the h/g function for each prediction set"""
        data_path = f"{config.ROOT_DIR}/data/{fn_name}_{self.score_type}/noise{config.noise_level}"
        file_path = f"{data_path}/{config.model_name}{'_pred_set' if config.HUMAN_DATASET=='PS' else ''}.csv"
        if not os.path.exists(file_path):
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            fn_per_set = []
            
            # Compute the h/g value given each human prediction 
            for image_name, participant_id, human_correct in self.human.itertuples(index=True):
                true_label = self.data.loc[image_name]["category"]
                model_score_true_label = self.model.loc[image_name][true_label]
                label_ranks = config.N_LABELS - self.model.drop(columns=['correct']).loc[image_name].argsort().argsort()
                true_label_rank = label_ranks[true_label]
                model_score_max = self.model.drop(columns=['correct']).loc[image_name].max()

                if "h" in fn_name:
                    fn_value_valid = 0
                    fn_value_invalid = human_correct
                else:
                    fn_value_valid = 1 - human_correct
                    fn_value_invalid = 0
                
                fn_per_set.append((image_name, participant_id, model_score_true_label, model_score_max, true_label_rank, fn_value_valid, fn_value_invalid))

            columns = ["image_name", "participant_id", "model_score_true_label", "model_score_max", "true_label_rank",f"{fn_name}_valid_sets", f"{fn_name}_invalid_sets"]            
            fn_df = pd.DataFrame(fn_per_set, columns=columns).set_index('image_name')  
            fn_df.to_csv(file_path)
        else:
            fn_df = pd.read_csv(file_path, index_col='image_name')

        return fn_df

    def set_data_set(self, data_set):
        self.data_set = data_set
        self.data_set_size = len(data_set)
        self.emp_risks = {}
        self.H_data_set = self.H.loc[self.data_set.index.values]
        self.G_data_set = self.G.loc[self.data_set.index.values]    
    
    def control(self):
        """Min control level per lambda for h and g"""
        n = self.data_set_size
        thresholds = (((n+1)*self.alphas - 1)/n)
        lamdas_dict = {np.around(lamda, decimals=config.lamda_dec):{} for lamda in np.arange(config.lambda_min,config.lambda_max+config.lambda_step,config.lambda_step)}

        for lamda in np.arange(config.lambda_min,config.lambda_max+config.lambda_step, config.lambda_step):
            emp_risk_lamda = self._empirical_risk(lamda)
            
            # Min alpha such that lambda is harm controlling under CF (Counterfactual) monotonicity
            min_alpha_idx_CF = np.searchsorted(thresholds, emp_risk_lamda, side='left')
        
            # For each lambda keep the min level of control under CF monotonicity
            lamdas_dict[np.around(lamda, decimals=config.lamda_dec)]['CF'] = np.round(self.alphas[min_alpha_idx_CF], decimals=2)

            # Empirical benefit (\hat G)
            emp_benefit_lamda = self._empirical_benefit(config.lambda_max-lamda)

            # Select smallest alpha that for which lambda is g controlling under cI (Interventional) monotonicity
            min_alpha_idx_cI = np.searchsorted(thresholds, emp_benefit_lamda, side='left')
            lamdas_dict[np.around(config.lambda_max - lamda, decimals=config.lamda_dec)]['cI'] = np.round(self.alphas[min_alpha_idx_cI], decimals=2)
        
        return lamdas_dict
       
    def compute(self):
        """Evaluate the counterfactual harm (bound)"""
        lamdas_dict = {np.around(lamda, decimals=config.lamda_dec):{} for lamda in np.arange(config.lambda_min,config.lambda_max+config.lambda_step,config.lambda_step)}

        for lamda in lamdas_dict.keys():
            harm_sum, harm_count = self._empirical_risk(lamda)
            g_harm, g_count = self._empirical_benefit(lamda)
            lamdas_dict[lamda]['hat_H'] = (harm_sum, harm_count)
            lamdas_dict[lamda]['hat_G'] = (g_harm, g_count)
        
        return lamdas_dict
    
    def tune_saps(self, x_y, run):
        """Find the optimal saps weight parameter value for each lamda value in a given run"""
        self.weights = {}
        weight_per_run_path = f"{config.results_path}/{config.model_name}/noise{config.noise_level}/saps/saps_weights_tune{config.tuning_split}_run{run}.json" 
        # Read the weights if already computed 
        if os.path.exists(weight_per_run_path):
            with open(weight_per_run_path, 'rt') as f:
                self.weights = json.load(f)
        else:
            self.weight_per_run = {}
            lamdas = np.array([lamda for lamda in np.arange(config.lambda_min,config.lambda_max+config.lambda_step,config.lambda_step)])
            min_avg_size_per_lambda = np.ones_like(lamdas)*config.N_LABELS
            self.weight_per_lamda = np.ones_like(lamdas)*config.SAPS_WEIGHTS[0] 
            tune_set_probs = self.model.drop(columns=['correct']).loc[x_y.index]
            tune_set_label_ranks = deepcopy(tune_set_probs)
            for idx in x_y.index:
                tune_set_label_ranks.loc[idx] = tune_set_probs.loc[idx].argsort()
            max_scores = tune_set_probs.max(axis=1)
            
            for weight in config.SAPS_WEIGHTS:
                # Get the saps scores in the tune set
                saps_scores_per_label = saps_batch(
                    weight,
                    tune_set_probs.loc[x_y.index.to_numpy()],
                    tune_set_label_ranks,
                    max_scores
                )
                # Compute the average prediction set size given a weight value
                set_sizes_per_lambda = []
                for idx in saps_scores_per_label.index:
                    set_sizes_per_lambda.append(np.searchsorted(
                        saps_scores_per_label.loc[idx],
                        lamdas,
                        side='right',
                        sorter=saps_scores_per_label.loc[idx].argsort()
                    ))
                set_sizes_per_lambda = np.stack(set_sizes_per_lambda, ).T
                set_sizes_per_lambda[set_sizes_per_lambda == 0] = 1
                avg_set_sizes_per_lambda = set_sizes_per_lambda.mean(axis=1)
                # Keep track of the weight value with the minimum average set size 
                self.weight_per_lamda[
                    avg_set_sizes_per_lambda < 
                    min_avg_size_per_lambda] = weight
                min_avg_size_per_lambda = np.minimum(avg_set_sizes_per_lambda, min_avg_size_per_lambda)
            # Save the optimal weight for each lambda 
            for i,lamda in enumerate(lamdas):
                lam_key = str(np.around(lamda, decimals=config.lamda_dec))
                self.weights[lam_key] = self.weight_per_lamda[i]
            create_path(f"{config.results_path}/{config.model_name}/noise{config.noise_level}/saps/")
            with open(weight_per_run_path, 'w') as f:
                json.dump(self.weights,f)