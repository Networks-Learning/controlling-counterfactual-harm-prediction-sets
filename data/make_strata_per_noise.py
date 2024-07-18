import config
from config import DATA_PATH
import pandas as pd
import os
from utils import create_path
from data.utils import stratify_images

"""Reads the datasets, stratify images and human 
    predictions per noise level, and compute the 
    mixture of MNLs per noise level. Saves the data
    under \data. Needs to run before main experiment"""

RAW_MODEL_DATA_PATH = f"{DATA_PATH}/hai_epoch10_model_preds_max_normalized.csv"
LABELS = [ "knife","keyboard","elephant","bicycle","airplane","clock","oven","chair","bear","boat","cat","bottle","truck","car","bird","dog"]
def single_model(name):
    """Create csv with all model outputs per image"""
    # Get all model predictions
    raw_data = pd.read_csv(RAW_MODEL_DATA_PATH, dtype=str)
    raw_model_predictions = raw_data.loc[raw_data["model_name"] == name]

    # Get all model predictions per noise level
    for noise_level in [80, 95, 110]:
        raw_model_pred_per_noise = raw_model_predictions.loc[raw_model_predictions["noise_level"] == str(noise_level)]
        model_predictions = raw_model_pred_per_noise[["image_name", "knife","keyboard","elephant","bicycle","airplane","clock","oven","chair","bear","boat","cat","bottle","truck","car","bird","dog", "correct"]]
        model_predictions.set_index("image_name", inplace=True)
        # Save model output
        model_pred_path = f"{DATA_PATH}/models/noise{noise_level}" 
        
        create_path(model_pred_path)
        
        model_predictions.to_csv(f"{model_pred_path}/{name}.csv")
    
def all_models():
    """Split the model predictions per model and noise level"""
    for model_name in ["alexnet", "vgg19", "densenet161", "googlenet", "resnet152"]:
        single_model(model_name)

def dataset():
    """Save the unique images and the true labels"""
    raw_data = pd.read_csv(RAW_MODEL_DATA_PATH, dtype=str)

    # Keep all the data once (fixed model and noise level)
    single_model_noise = raw_data.loc[(raw_data["model_name"] == "vgg19") & (raw_data["noise_level"] == "80")]

    # Isolate images and the true labels
    x_str_y = single_model_noise[["image_name", "category"]]
    x_str_y.set_index("image_name", inplace=True)

    # Save data and labels
    x_str_y.to_csv(f"{DATA_PATH}/imagenet16H.csv") 

def human_successes():
    """Human success or fail when deciding alone for each noise level"""
    path = f"{DATA_PATH}/human_only_classification_6per_img_export.csv"
    raw_data = pd.read_csv(path, dtype=str)
    
    # Get all human predictions per noise level
    for noise_level in [80, 95, 110]:
        raw_pred_per_noise = raw_data.loc[raw_data["noise_level"] == str(noise_level)]
   
        # Keep image_name, participant id and success or fail
        image_human_correct = raw_pred_per_noise[["image_name", "participant_id", "correct"]].set_index("image_name")

        path = f"{DATA_PATH}/users/noise{noise_level}"
        create_path(path)
        # Save the data
        image_human_correct.to_csv(f"{path}/success.csv")

def human_predictions():
    """Human prediction when deciding alone for each noise level"""
    path = f"{DATA_PATH}/human_only_classification_6per_img_export.csv"
    raw_data = pd.read_csv(path, dtype=str)
    
    # Get all human predictions per noise level
    for noise_level in [80, 95, 110]:
        raw_pred_per_noise = raw_data.loc[raw_data["noise_level"] == str(noise_level)]
   
        # Keep image_name, participant id and prediction
        image_human_correct = raw_pred_per_noise[["image_name", "participant_id", "participant_classification"]].set_index("image_name")

        path = f"{DATA_PATH}/users/noise{noise_level}"
        create_path(path)
        # Save the data
        image_human_correct.to_csv(f"{path}/predictions.csv")

def human_success_pred_sets():
    """Human success or fail for the noise level 110 when deciding alone 
       from the study with prediction sets (ImageNet16H-PS)"""
    path = f"{DATA_PATH}/study_pred_sets/rewards_strict.csv"
    raw_data = pd.read_csv(path)
    
    raw_human_pred_alone = raw_data.loc[raw_data["set"] == 16]

    # Keep image_name, worker id and success or fail
    image_human_correct = raw_human_pred_alone[["image_name", "worker_id", "reward"]].set_index("image_name")
    image_human_correct['reward'] = image_human_correct['reward'].astype(int)

    path = f"{DATA_PATH}/users/noise110"
    create_path(path)
    # Save the data
    image_human_correct.to_csv(f"{path}/success_pred_set.csv")
    
def model_sorted_predictions(name):
    """Create csv with with the model prediction and all
    labels for each image sorted based on the model's output"""
    dtypes = {
        'image_name':str,
        'model_pred':str,
        "noise_level":str
        }
    for l in LABELS:
        dtypes[l] = float

    # Get all model info
    raw_data = pd.read_csv(RAW_MODEL_DATA_PATH, dtype=dtypes)
    raw_model_predictions = raw_data.loc[raw_data["model_name"] == name]

    # Get all model outputs and predictions per noise level
    for noise_level in [80, 95, 110]:
        raw_model_out_pred_per_noise = raw_model_predictions.loc[raw_model_predictions["noise_level"] == str(noise_level)]
        model_output_predictions = raw_model_out_pred_per_noise[["image_name"]+LABELS+["model_pred"]]
        model_output_predictions.set_index("image_name", inplace=True)

        # Sort labels per output value in ascending order
        idx = model_output_predictions[LABELS].values.argsort(axis=1)
        # Reverse the order
        sorted_labels = pd.DataFrame(
            model_output_predictions.columns.to_numpy()[idx[::,::-1]],
            index=model_output_predictions.index
            )
        model_output_predictions[LABELS] = sorted_labels
        # Rename former label columns with label rank
        model_output_predictions.rename(
            {l: num+1 for num, l in enumerate(LABELS)},
            axis='columns',
            inplace=True
            )

        # Save model predictions
        model_pred_path = f"{DATA_PATH}/models/noise{noise_level}" 
        
        create_path(model_pred_path)
        
        model_output_predictions.to_csv(f"{model_pred_path}/{name}_sorted.csv")
   
def all_models_sorted():
    """Split and sort the model predictions per model and noise level"""
    for model_name in ["alexnet", "vgg19", "densenet161", "googlenet", "resnet152"]:
        model_sorted_predictions(model_name)

def mnl_human_ps(n_stratas_images=2, model='vgg19', noise_level=110):
    """For each image and prediction set compute the human success
    probability using an MNL model per image difficulty level (image_strata)
    using the ImageNet16H-PS dataset"""

    path = f"{DATA_PATH}/study_pred_sets/predictions_strict.csv"
    all_human_predictions = pd.read_csv(path, dtype=str, index_col='image_name')
        
    success_prob_path = f"{DATA_PATH}/users/noise{noise_level}/model{model}"
    create_path(success_prob_path)

    # Read human predictions
    human_alone = all_human_predictions.loc[all_human_predictions["set"] == str(config.N_LABELS)]

    # Read images true labels
    x_y = pd.read_csv(f"{DATA_PATH}/imagenet16H.csv", dtype=str, index_col='image_name')

    human_alone['image_category'] = x_y['category']

    human_alone = human_alone.rename(
            {
                'worker_id':'participant_id', 
                'prediction':'participant_classification'
            }, 
            axis=1
        ).reset_index()
    # Read labels sorted by the model output
    model_pred_path = f"{DATA_PATH}/models/noise{noise_level}" 
    
    model_pred_sorted = pd.read_csv(f"{model_pred_path}/{model}_sorted.csv", dtype=str, index_col='image_name')

    # Keep image_name, participant id, true label and prediction
    image_human_true_pred = human_alone[["image_name", "participant_id", "image_category","participant_classification"]].set_index("image_name")

    # Stratify images per difficulty (equal sized stratas)
    image_stratas = stratify_images(noise_level=noise_level, n_stratas_images=n_stratas_images)

    image_set_reward_prob = []
    for i, strata in enumerate(image_stratas):
        
        # Get human predictions per strata
        strata_human_true_pred = image_human_true_pred.loc[strata]

        # Compute the confusion matrix for each strata            
        conf_matrix = pd.crosstab(
            strata_human_true_pred['image_category'],
            strata_human_true_pred['participant_classification']
            )
        # Normalized confusion matrix
        conf_matrix_norm = conf_matrix.div(conf_matrix.sum(axis=1), axis='index')
        
        for image in strata.values:
            # True label
            y = strata_human_true_pred.loc[image]['image_category'].values[0]
            
            # Nominator of mnl model
            c_yy = 0
            
            # Denominator of mnl model
            mnl_denom = 0

            for set_size in range(1, 17):
                label = model_pred_sorted.loc[image][str(set_size)]

                if label == y:
                    c_yy = conf_matrix_norm.loc[y][y]

                mnl_denom+=conf_matrix_norm.loc[y][label]
                
                if c_yy == 0:
                    success_prob = 0
                else:
                    success_prob = c_yy / mnl_denom

                image_set_reward_prob.append((image, set_size, success_prob))
        
    strata_image_set_reward_prob = pd.DataFrame(image_set_reward_prob, columns=['image_name', 'set', 'reward'])

    strata_image_set_reward_prob.to_csv(f"{success_prob_path}/success_prob_stratas{n_stratas_images}_PS.csv", index=False)

def mnl_human(n_stratas_images=1, model='vgg19'):
    """For each image and prediction set compute the human success
    probability using an MNL model per image difficulty level (image_strata)
    using the ImageNet16H dataset"""

    path = f"{DATA_PATH}/human_only_classification_6per_img_export.csv"
    raw_data = pd.read_csv(path, dtype=str)
    
    # Get all human predictions per noise level
    for noise_level in [80, 95, 110]:
        
        success_prob_path = f"{DATA_PATH}/users/noise{noise_level}/model{model}"
        create_path(success_prob_path)

        # Read human predictions
        raw_pred_per_noise = raw_data.loc[raw_data["noise_level"] == str(noise_level)]
   
        # Read labels sorted by the model output
        model_pred_path = f"{DATA_PATH}/models/noise{noise_level}" 
        
        model_pred_sorted = pd.read_csv(f"{model_pred_path}/{model}_sorted.csv", dtype=str, index_col='image_name')
   
        # Keep image_name, participant id, true label and prediction
        image_human_true_pred = raw_pred_per_noise[["image_name", "participant_id", "image_category","participant_classification"]].set_index("image_name")

        # Stratify images per difficulty (equal sized stratas)
        image_stratas = stratify_images(noise_level=noise_level, n_stratas_images=n_stratas_images)

        image_set_reward_prob = []
        for i, strata in enumerate(image_stratas):
            
            # Get human predictions per strata
            strata_human_true_pred = image_human_true_pred.loc[strata]

            # Compute the confusion matrix for each strata            
            conf_matrix = pd.crosstab(
                strata_human_true_pred['image_category'],
                strata_human_true_pred['participant_classification']
                )
            # Normalized confusion matrix
            conf_matrix_norm = conf_matrix.div(conf_matrix.sum(axis=1), axis='index')
            
            for image in strata.values:
                # True label
                y = strata_human_true_pred.loc[image]['image_category'].values[0]
                
                # Nominator of mnl model
                c_yy = 0
                
                # Denominator of mnl model
                mnl_denom = 0

                for set_size in range(1, 17):
                    label = model_pred_sorted.loc[image][str(set_size)]

                    if label == y:
                        c_yy = conf_matrix_norm.loc[y][y]

                    mnl_denom+=conf_matrix_norm.loc[y][label]
                    
                    if c_yy == 0:
                        success_prob = 0
                    else:
                        success_prob = c_yy / mnl_denom

                    image_set_reward_prob.append((image, set_size, success_prob))
            
        strata_image_set_reward_prob = pd.DataFrame(image_set_reward_prob, columns=['image_name', 'set', 'reward'])

        strata_image_set_reward_prob.to_csv(f"{success_prob_path}/success_prob_stratas{n_stratas_images}.csv", index=False)

def all_mnl():
    """Compute the mixture of MNLs for each model and noise level"""
    for model_name in ["alexnet", "vgg19", "densenet161", "googlenet", "resnet152"]:
        for n_str in [2]:
            mnl_human(n_stratas_images=n_str, model=model_name)


if __name__=="__main__":
    dataset()
    all_models()
    human_successes()
    human_predictions()
    human_success_pred_sets()
    all_models_sorted()
    mnl_human_ps(n_stratas_images=2, model='vgg19', noise_level=110)
    all_mnl()