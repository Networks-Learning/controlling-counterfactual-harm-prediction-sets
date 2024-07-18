import subprocess
from copy import deepcopy

"""Execute scripts.counterfactual_harm for all noise levels and models"""

args_base = ["python", "-m", "scripts.counterfactual_harm", "--mode", "control"]
models = ["alexnet", "vgg19", "densenet161", "googlenet", "resnet152"]

for model in models:
    args = deepcopy(args_base)
    args.append("--model_name")
    args.append(model)
    for noise_level in [80, 95, 110]:
        args.append("--noise_level")
        args.append(str(noise_level))
        subprocess.run(args=args)    


    
    