import os
import numpy as np
import argparse

"""Experiments configuration"""

ROOT_DIR = os.path.dirname(__file__)
DATA_PATH = f"{ROOT_DIR}/data"

N_RUNS = 50
DEBUG = False
N_LABELS = 16
SAPS_WEIGHTS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, \
                    choices=["alexnet", "vgg19", "densenet161", "googlenet", "resnet152"], \
                    help='Choose the model used by the set valued predictor.', \
                    default='vgg19')
parser.add_argument("--noise_level", choices=[80, 95, 110, 125], type=int, \
                    help='Choose the noise level applied in the images. \
                    Datatsets are available only for the following noise levels 80, 95, 110, and 125',\
                    default=110)
parser.add_argument("--alpha", type=float, default=0.1,\
                    help="The desired risk controlling level")
parser.add_argument("--mode", choices=['control', 'test'], type=str,\
                    help="Choose the operating mode: control (finding the harm controlling lambdas) or\
                         test (computing accuracy and harm over test set).",
                    default='control')
parser.add_argument("--dataset", choices=["PS",""],type=str,\
                    help="Set to PS for experiments using the \
                    ImageNet16H-PS dataset. Set only for \
                    model vgg19 and noise level 110.",\
                    default="")
parser.add_argument("--mnl_ps", choices=[True, False],\
                    type=bool, help="Set to True to run experiments\
                    with a mixture of MNLs using the ImageNet16H-PS data",
                    default=False)
parser.add_argument("--score_type", choices=["vanilla", "SAPS"], \
                    default="vanilla", type=str,
                    help="Choose the label score function type (non-conformity score)\
                    to select the labels in the prediction set.")

args,unknown = parser.parse_known_args()

# Operating mode (test or control)
mode = args.mode

# The classifier used to produce the prediction sets
model_name = args.model_name

# The noise level applied to the images (higher noise, more difficult task)
noise_level = args.noise_level 

# Set harm control level
alpha = args.alpha

# Granularity of harm level grid
alpha_step = 0.01 

# Select dataset
HUMAN_DATASET = args.dataset

# Select data for MNL mixture
MNL_PS = args.mnl_ps

# Initialize random generators
entropy = 0x3034c61a9ae04ff8cb62ab8ec2c4b501
numpy_rng = np.random.default_rng(entropy)
tune_rng = np.random.default_rng(entropy)
saps_rng = np.random.default_rng(entropy)

# Fraction of the dataset to use as calibration set
calibration_split = 0.1

# Fraction of the dataset to use as a tuning set for SAPS
tuning_split = 0.1

# Name of score function (non-conformity score)
score_type = args.score_type

# Range of lambda values according to score type
lambda_min = 0. 
lambda_max = 1. if score_type == 'vanilla' else (1. + SAPS_WEIGHTS[-1]*(N_LABELS-1))

# The granularity for the lambda grid
lambda_step = (lambda_max - lambda_min) / 1000

# Decimals to round up spurious lambda values due to np.arange
lamda_dec = 3 if score_type == 'vanilla' else 5

# Set up paths for results and plots
# Path to store results
results_path = f"results" 

# Path to store plots
plot_path = f"plots"
