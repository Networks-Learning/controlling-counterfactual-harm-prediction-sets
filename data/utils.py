from plotters.interventional import thresholds_per_strata, get_strata
import config
import pandas as pd
import numpy as np


def stratify_images(noise_level=110, n_stratas_images=1):
    """Stratify images of a given noise level into levels of difficulty"""

    # Read succeses of humans alone
    rewards_file_path = f"{config.ROOT_DIR}/data/users/noise{noise_level}/success.csv"
    rewards = pd.read_csv(rewards_file_path, dtype={'correct':np.int16}).set_index('image_name')

    rewards.rename({'correct':'reward'}, axis='columns', inplace=True)
    
    images_acc_thr_per_strata, avg_acc_per_image = thresholds_per_strata('image_name', rewards, n_stratas_images)
    image_stratas = []
    images_low_threshold = 0
    for j in range(1, n_stratas_images+1):
        is_last = j == n_stratas_images
        # Get images of difficulty strata
        strata_images = get_strata(j, images_low_threshold, avg_acc_per_image, images_acc_thr_per_strata, is_last)
        image_stratas.append(strata_images)
        
        images_low_threshold = images_acc_thr_per_strata[j]["high"]

    return image_stratas