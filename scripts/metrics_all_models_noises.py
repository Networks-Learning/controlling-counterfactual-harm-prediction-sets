import subprocess
from copy import deepcopy

"""Computes average accuracy for all models and noise levels"""
# Change "vanilla" to "SAPS" for results using the SAPS non-conformity score
score_type = "vanilla"
args_base = ["python", "-m", "scripts.test_metrics", "--mode", "test", "--score_type", score_type]
models = ["alexnet", "vgg19", "densenet161", "googlenet", "resnet152"]

for model in models:
    args = deepcopy(args_base)
    args.append("--model_name")
    args.append(model)
    for noise_level in [80, 95, 110]:
        args.append("--noise_level")
        args.append(str(noise_level))
        subprocess.run(args=args) 
        if noise_level == 110 and model == "vgg19" and score_type == "vanilla":
            # Estimate accuracy using the real predictions from ImageNet16H-PS  
            args_real = deepcopy(args)
            args_real.append("--dataset")   
            args_real.append("PS")   
            subprocess.run(args=args_real)

            # Estimate accuracy using a mixture of MNLs computed 
            # with the prediction of experts on their own 
            # in ImageNet16H-PS
            args_mnl_ps = deepcopy(args)
            args_mnl_ps.append("--mnl_ps") 
            args_mnl_ps.append("True") 
            subprocess.run(args=args_mnl_ps)