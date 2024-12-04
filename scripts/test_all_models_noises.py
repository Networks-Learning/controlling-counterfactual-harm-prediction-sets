import subprocess
from tqdm import tqdm
from copy import deepcopy
import numpy as np

"""Executes scripts.counterfactual_harm at test time for all models and noise levels"""
# Change score_type to "SAPS" for results using the SAPS non-conformity score
score_type = "vanilla"
args_base = ["python", "-m", "scripts.counterfactual_harm", "--mode", "test", "--score_type", score_type]
models = ["alexnet", "vgg19", "densenet161", "googlenet", "resnet152"]

for model in models:
    args = deepcopy(args_base)
    args.append("--model_name")
    args.append(model)
    for noise_level in [80, 95, 110]:
        args.append("--noise_level")
        args.append(str(noise_level))
        subprocess.run(args=args)    


    
    