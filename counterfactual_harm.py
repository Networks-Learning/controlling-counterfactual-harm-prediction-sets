import config 
import numpy as np
import os
import pandas as pd

"""Main module to control and evaluate counterfactual harm (bound)"""

class CounterfactualHarm:
    def __init__(self, model, human, data) -> None:
        self.model = model
        self.human = human
        self.data = data
        self.emp_risks = {}
        self.H = self.fn_per_set("h")
        self.G = self.fn_per_set("g")
        self.alphas = np.arange(0,1+config.alpha_step, config.alpha_step)

    def _empirical_risk(self, lamda):
        """Computes hat H(\lambda)"""
        # Set is valid or corner case of empty set
        is_valid_set = (self.H_data_set['model_score_true_label'] >= (1 - lamda)) | (self.H_data_set['model_score_max'] == self.H_data_set['model_score_true_label'])
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
        is_valid_set = (self.G_data_set['model_score_true_label'] >= (1 - lamda)) | (self.H_data_set['model_score_max'] == self.H_data_set['model_score_true_label'])

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
        data_path = f"{config.ROOT_DIR}/data/{fn_name}/noise{config.noise_level}"
        file_path = f"{data_path}/{config.model_name}{'_pred_set' if config.HUMAN_DATASET=='PS' else ''}.csv"
        if not os.path.exists(file_path):
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            fn_per_set = []
            # Compute the h/g value given each human prediction 
            for image_name, participant_id, human_correct in self.human.itertuples(index=True):
                true_label = self.data.loc[image_name]["category"]
                model_score_true_label = self.model.loc[image_name][true_label]
                
                model_score_max = self.model.drop(columns=['correct']).loc[image_name].max()

                if fn_name == "h":
                    fn_value_valid = 0
                    fn_value_invalid = human_correct
                else:
                    fn_value_valid = 1 - human_correct
                    fn_value_invalid = 0
                
                fn_per_set.append((image_name, participant_id, model_score_true_label, model_score_max, fn_value_valid, fn_value_invalid))

            columns = ["image_name", "participant_id", "model_score_true_label", "model_score_max", f"{fn_name}_valid_sets", f"{fn_name}_invalid_sets"]
            
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
        lamdas_dict = { np.around(lamda, decimals=3):{} for lamda in np.arange(0,1+config.lambda_step,config.lambda_step)}

        for lamda in np.arange(0,1+config.lambda_step, config.lambda_step):
            emp_risk_lamda = self._empirical_risk(lamda)
            
            # Min alpha such that lambda is harm controlling under CF (Counterfactual) monotonicity
            min_alpha_idx_CF = np.searchsorted(thresholds, emp_risk_lamda, side='left')
        
            # For each lambda keep the min level of control under CF monotonicity
            lamdas_dict[np.around(lamda, decimals=3)]['CF'] = np.round(self.alphas[min_alpha_idx_CF], decimals=2)

            # Empirical benefit (\hat G)
            emp_benefit_lamda = self._empirical_benefit(1.-lamda)

            # Select smallest alpha that for which lambda is g controlling under cI (Interventional) monotonicity
            min_alpha_idx_cI = np.searchsorted(thresholds, emp_benefit_lamda, side='left')
            lamdas_dict[np.around(1 - lamda, decimals=3)]['cI'] = np.round(self.alphas[min_alpha_idx_cI], decimals=2)
        
        return lamdas_dict
       
    def compute(self):
        """Evaluate the counterfactual harm (bound)"""
        lamdas_dict = { lamda:{} for lamda in np.arange(0,1+config.lambda_step,config.lambda_step)}

        for lamda in np.arange(0,1+config.lambda_step, config.lambda_step):
            harm_sum, harm_count = self._empirical_risk(lamda)
            g_harm, g_count = self._empirical_benefit(lamda)
            lamdas_dict[lamda]['hat_H'] = (harm_sum, harm_count)
            lamdas_dict[lamda]['hat_G'] = (g_harm, g_count)
        
        return lamdas_dict