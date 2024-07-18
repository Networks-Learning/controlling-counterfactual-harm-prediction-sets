import config 
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import create_path

"""Produces the plots showing average accuracy per 
prediction set size for prediction sets that include
the ground truth label"""

mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts,geometry}'
mpl.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams.update({
    'font.family':'serif',
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
    "font.size": 48,
    "figure.figsize":(16,10)
})


def read_rewards():
    """Returns the rewards for each image and prediction set"""
    rewards_path = f"{config.ROOT_DIR}/data/study_pred_sets/rewards_strict.csv"
    rewards = pd.read_csv(rewards_path, dtype={'set':np.int32, 'reward':np.bool_}).set_index('image_name')
    return rewards

data_path = f"{config.ROOT_DIR}/data/hai_epoch10_model_preds_max_normalized.csv"
IMAGE_TRUE_LABEL = pd.read_csv(data_path, header=0, index_col='image_name')
unique_label_values = np.unique(IMAGE_TRUE_LABEL['category'].values)
LABEL = {i:v for i,v in enumerate(unique_label_values)}
# {0: 'airplane', 1: 'bear', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'car', 7: 'cat', 8: 'chair', 9: 'clock', 10: 'dog', 11: 'elephant', 12: 'keyboard', 13: 'knife', 14: 'oven', 15: 'truck'} 

def get_reward_valid_sets(label, setting='strict'):
    """Returns the rewards for the prediction sets 
    that include the true label"""
    rewards = read_rewards(setting=setting)
    
    
    columns_to_keep = ['set', 'reward', 'worker_id']
    
    
    set_reward_df = rewards[columns_to_keep]

    
    for worker_to_drop in [f"y{i}" for i in range(1,6)]:
         set_reward_df = set_reward_df.where(set_reward_df['worker_id']!=worker_to_drop)
         set_reward_df.dropna(inplace=True)
    
    reward_valid_sets = set_reward_df
    
    images_per_label = IMAGE_TRUE_LABEL[IMAGE_TRUE_LABEL['category']==LABEL[label]].index
    reward_valid_sets_per_label = reward_valid_sets.loc[images_per_label]
    reward_valid_sets_per_label['reward'] = reward_valid_sets_per_label['reward'].astype(int)
    
    return reward_valid_sets_per_label

def plot(reward_valid_sets, ylim_low=None, ylim_high=None, n_image_strata=0, n_worker_strata=0):
    """Plots the empirical success probability per prediction set size
    for the prediction sets that include the true label"""
    reward_valid_sets = reward_valid_sets[reward_valid_sets['set'] > 1]
    errors = (reward_valid_sets.groupby('set').std() / np.sqrt(reward_valid_sets.groupby('set').count()))
    print(reward_valid_sets)
    acc_per_set = reward_valid_sets.groupby('set').mean()
    print(acc_per_set)
    ax = acc_per_set.plot.bar(yerr=errors, rot=0)
    low_lim = .02 if n_image_strata < 3 else .002
    high_lim = 0.01 if n_image_strata < 3 else .001
    ylim_low = min(acc_per_set.values - errors.values) - 0.01
    ylim_high = max(acc_per_set.values + errors.values) 
    
    # Adjust the scale by setting y_lims 
    if n_image_strata >= 2 and n_worker_strata == 2:
        ax.set_ylim(bottom=0.5, top=1.)
    else:
        ax.set_ylim(bottom=ylim_low, top=ylim_high)

    ax.set_xlabel(r"Prediction Set Size")
    ax.set_ylabel(r"Empirical Success Probability")
    ax.tick_params(axis='y', width=4, length=20)
    ax.spines[['right', 'top']].set_visible(False)
    ax.get_legend().remove()

def thresholds_per_strata(key, reward_valid_sets, n_stratas):    
    """Compute the threshold value for each strata of <key>,
    where <key> can be one of {'image_name', 'worker_id'}"""
  
    avg_acc_per_key = reward_valid_sets.groupby(key).mean(numeric_only=True)['reward'].to_frame()
    

    acc_thresholds_per_strata = {}
    for i in range(1, n_stratas+1):        
        acc_thresholds_per_strata[i] = {
            "high": avg_acc_per_key.quantile((1.*i)/n_stratas, axis=0)['reward']
        }
 
    return acc_thresholds_per_strata, avg_acc_per_key

def get_strata(n_strata, low_threshold, avg_acc_per_key, acc_thr_per_strata, is_last=False):
    """Returns the strata of <key>, that is images of similar 
    difficulty or workers with the same level of competence"""
    lb_q = avg_acc_per_key['reward'] >= low_threshold
    ub_q = avg_acc_per_key['reward'] < acc_thr_per_strata[n_strata]["high"]
    if is_last:
        ub_q = avg_acc_per_key['reward'] <= acc_thr_per_strata[n_strata]["high"]

    strata = avg_acc_per_key[(lb_q)&(ub_q)].index
    return strata

def per_strata(label,n_stratas_workers=1, n_stratas_images=1, strata_to_plot_workers=0, strata_to_plot_images=0):
    """Stratify dataset with respect to the difficulty of the images 
    and the competence level of workers and plot the average accuracy
    per prediction set size for prediction sets that include the 
    true label for each strata of difficulty."""
    def plot_strata(reward_valid_sets, worker_strata_n, image_strata_n, workers_total_stratas, images_total_stratas):
        plot(reward_valid_sets, n_image_strata=image_strata_n, n_worker_strata=worker_strata_n)
        worker_dir_path = f"{config.ROOT_DIR}/{config.plot_path}/monotonicity/workers{workers_total_stratas}/label_{label}/"
        create_path(worker_dir_path)

        worker_image_path = worker_dir_path + f"images{images_total_stratas}/"
        if not os.path.exists(worker_image_path):
            os.mkdir(worker_image_path)

        plot_path = worker_image_path + f"workers_{worker_strata_n}_images_{image_strata_n}_nosingleton.pdf"

        plt.savefig(plot_path, bbox_inches='tight')
        
    reward_valid_sets = get_reward_valid_sets(label)
    workers_acc_thr_per_strata, avg_acc_per_worker = thresholds_per_strata('worker_id', reward_valid_sets, n_stratas_workers)
    
    images_acc_thr_per_strata, avg_acc_per_image = thresholds_per_strata('image_name', reward_valid_sets, n_stratas_images)
    workers_low_threshold =  0
    for i in range(1, n_stratas_workers+1):
        last_workers = i == n_stratas_workers
        # Get workers strata
        strata_workers = get_strata(i, workers_low_threshold, avg_acc_per_worker, workers_acc_thr_per_strata, last_workers)
        print(strata_workers)
        images_low_threshold = 0
        for j in range(1, n_stratas_images+1):
            is_last = j == n_stratas_images
            # Get images strata
            strata_images = get_strata(j, images_low_threshold, avg_acc_per_image, images_acc_thr_per_strata, is_last)
            workers_q = reward_valid_sets['worker_id'].isin(strata_workers)
            images_q = reward_valid_sets.index.isin(strata_images)
            
            strata = reward_valid_sets.loc[(workers_q)&(images_q)][['set', 'reward']]
            strata = strata.astype({'set': 'int16'})
            print(strata[strata['set']==1])
            
            plot_all = strata_to_plot_workers == 0 and strata_to_plot_images == 0
            plot_this_strata = strata_to_plot_workers == i and strata_to_plot_images == j
            if plot_all or plot_this_strata:
                plot_strata(strata, worker_strata_n=i, image_strata_n=j, workers_total_stratas=n_stratas_workers, images_total_stratas=n_stratas_images)
        
            images_low_threshold = images_acc_thr_per_strata[j]["high"]

        workers_low_threshold = workers_acc_thr_per_strata[i]["high"]

if __name__=="__main__":
    """Produces the plots of the average accuracy against prediction set size
    for given ground truth labels across all experts and across experts 
    of different level of competence"""
    for n_stratas_workers in [1,2]:
        # To plot all the labels uncomment the following
        # for label in range(config.N_LABELS):
        for label in [4,11]:
            per_strata(label=label, n_stratas_images=4, n_stratas_workers=n_stratas_workers)
