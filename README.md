# Controlling Counterfactual Harm in Decision Support Systems Based on Prediction Sets

This repository contains the code used in the paper "Controlling Counterfactual Harm in Decision Support Systems Based on Prediction Sets", accepted at the ICML 2024 Workshop on Humans, Algorithmic Decision-Making and Society.

## **Install Dependencies**

Experiments ran on python 3.10.9. To install the required libraries run:

```pip install -r requirements.txt```


## **Code Structure**

### **Data**
The folder ```data/``` includes the ImageNet16H dataset and the ```data/study_pred_sets``` includes the ImageNet16H-PS dataset. It also includes the following scripts to pre-process the datasets for the experiments:
*  `make_strata_per_noise.py` stratifies the images, classifier and human predictions for each amount of noise $\omega\in \{80,95,110\}$ and stores the stratified datasets under `/data`. It also computes for each pair of images and prediction sets the predicted average accuracy of human experts using the mixture of MNL models.   
* `utils.py` includes a helper function that stratifies images with the same amount of noise into two different levels of difficulty. 

### **Setup**
* ```config.py``` includes all configuration parameters of the experimental setup.
* ```utils.py```  implements helper functions for saving results.
* ```counterfactual_harm.py``` implements the main module to control and evaluate counterfactual harm (bound) under the counterfactual and interventional monotonicity assumptions for a given classifier, noise level and calibration set.


### **Scripts for Running Experiments and Evaluation**
The folder ```scripts/``` includes the following scripts to execute the experiments:
* ```counterfactual_harm.py``` (`control` mode) computes for each control level $\alpha$ from 0 to 1 with step $0.01$ the harm controlling $\lambda$ values given by Corollary 1 and all controlling $\lambda$ values that are $\leq \check{\lambda}(\alpha)$ given by Thm 2 (combining these values with the ones of Corolarry 1 we have the harm controlling values of Corolarry 2), for 50 random samplings of the test and calibration set, given a fixed classifier and noise level. 
* ```counterfactual_harm.py``` (`test` mode) computes the average counterfactual harm (bound) for each $\lambda$ value across the 50 random samplings of the test set for a given classifier and  noise level.
* ```control_all_models_noises.py``` executes `scripts/counterfactual_harm.py` for all classifiers and noise levels under `control` mode.
* `test_all_models_noises.py` executes `scripts/counterfactual_harm.py` for all classifiers and noise levels under `test` mode.
* `test_metrics.py` estimates the average accuracy per $\lambda$ value 
 for each one of the 50 random samplings of the test set for a given classifier and noise level.  
* `metrics_all_models_noises.py` executes `scripts/test_metrics.py` for each classifier and noise level.

### **Plots**
```plotters/``` includes the following scripts to produce the plots in the paper:
* ```interventional.py``` produces the plots related to the average accuracy  per prediction set size for images of similar difficulty across all experts and across experts with the same level of competence for each ground truth label.
* ```harm_vs_acc.py``` produces the plots of average counterfactual harm against average accuracy for all noise levels and classifiers.  

## **Running Instructions**

### **Run Experiments**
To create the stratified datasets per noise level run:

```python -m data.make_strata_per_noise```

To compute all controlling $\lambda$ values (for each control level $\alpha$) for each of the 50 samplings of the calibration set for each classifier and noise level run:

```python -m scripts.control_all_models_noises```

To evaluate the average counterfactual harm (bound) for each $\lambda$ value for each one of the 50 samplings of the test set, and every classifier and noise level run:

```python -m scripts.test_all_models_noises```

To compute the average accuracy (predicted using the mixture of MNLs and real using ImageNet16H-PS dataset) for each $\lambda$ value for each one of the 50 samplings of the test set, and every classifier and noise level run:

```python -m scripts.metrics_all_models_noises```

All results will be saved under the folder `results`.

### **Plots** 
To produce the plots of average counterfactual harm against average accuracy for all noise levels and classifiers (Figures 2-5, 7-9) run

 ```python -m plotters.harm_vs_acc``` 

To produce the plots related to the average accuracy  per prediction set size for images of similar difficulty across all experts and across experts with the same level of competence for each ground truth label run (Figure 6):

```python -m plotters.interventional```

All plots will be saved under the folder `plots`.


## **Citation**
If you use parts of the code/data in this repository for your own research purposes, please consider citing:

```
@article{straitouri2024controlling,
  title={Controlling Counterfactual Harm in Decision Support Systems Based on Prediction Sets},
  author={Straitouri, Eleni and Thejaswi, Suhas and Rodriguez, Manuel Gomez},
  journal={arXiv preprint arXiv:2406.06671},
  year={2024}
}

```
