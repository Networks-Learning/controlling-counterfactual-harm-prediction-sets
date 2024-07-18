from utils import create_path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import config
from plotters.post_process_results import *
import os
import numpy as np

"""Produces the plots showing average accuracy against average counterfactual harm"""

mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts,geometry}'
mpl.rcParams['axes.formatter.use_mathtext'] = True
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=sns.color_palette('colorblind')) 
plt.rcParams.update({
    'font.family':'serif',
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
    "font.size": 48,
    "figure.figsize":(16,17.5), # figsize for stacked plots
    # "figure.figsize":(16,10), # figsize for simple plots
})

MARKERS = {
    'CF': 'o',
    'cI': 'x'
}
LOCATION = {
    'CF': 'upper center',
    'cI': 'upper center'
}
MODEL_TICKS = {
    r'VGG19': [0.8, 0.9],
    r'DenseNet161': [0.8, 0.9],
    r'ResNet152':[0.75, 0.85],
    r'GoogleNet':[0.75, 0.85],
    r'AlexNet':[0.7, 0.8],
}
MARKERS_MODELS = {
    r'VGG19': 'o',
    r'DenseNet161': '|',
    r'ResNet152':'x',
    r'GoogleNet':'p',
    r'AlexNet':'D',
    r'Real':'o',
    r'Predicted':'x'
}
COLORS_MODELS = {
    r'VGG19': 'C4',
    r'DenseNet161': 'C0',
    r'ResNet152':'C1',
    r'GoogleNet':'C2',
    r'AlexNet':'C3',
    r'Real': 'C0',
    r'Predicted': 'C1'
}

COLORS_NOISES = {
    80: 'C0',
    95: 'C1',
    110:'C2'
}

MARKERS_NOISES = {
    80: 'o',
    95: 'x',
    110:'p'
}
MAP_ASSUM_ALPHA = {
    'CF': {
        110: [0.01, 0.05],
        80:[0.01, 0.05],
        95:[0.01, 0.05],
    },
    'cI': {
        80: [0.12, 0.14],
        95: [0.15, 0.17],
        110: [.23, .24]
    }
}
AX_LABELS={
    'harm':r'Average Counterfactual Harm',
    'accuracy':r'Average Accuracy'
}

def harm_vs_metric_per_model(metric='accuracy',
                   xaxis='accuracy', 
                   yaxis='harm', 
                   models=['DensNet161'],
                   pred_sets=True, 
                   assumptions=['CF', 'cI'], 
                   show_alpha=True,
                   alpha=.1,
                   save=True,
                   ):
    """Plots and saves the real and predicted average accuracy
    usign the ImageNet16H-PS data against the average counterfactual 
    harm for a fixed alpha"""
    def confidence95(mean, se):
        low = mean - 1.96*se
        up = mean + 1.96*se
        assert all(1.96*se < 0.01)
        return low, up
    

    cmap = plt.colormaps['Blues']

    min_val = defaultdict(lambda:1.)
    max_val = defaultdict(lambda:0.)
    
    if pred_sets:
        models = ['Real', 'Predicted']

    for i, model_name in enumerate(models):
    # set metric to 'metrics' when pred_sets is True
        metr = 'metrics' if pred_sets else 'metric_2_PS'
        harm_metrics_lamda_freq = metric_vs_harm(
            pred_sets=pred_sets, 
            metric=metr, 
            model_name=model_name,
            alpha=alpha,
        )
        
        results_dict = {}
        for assum in assumptions:
            for m in ['harm', metric]:
                for val in ['mean', 'se']:
                    results_dict[(assum, m, val)] = harm_metrics_lamda_freq.loc[assum][f"{m}_{val}"].values
        
        if metric == 'harm':
            for assum in assumptions:
                results_dict[(assum, 'lambda', 'mean')] = harm_metrics_lamda_freq.loc[assum].index.to_numpy()
                results_dict[(assum, 'lambda', 'se')] = np.zeros_like(results_dict[(assum, 'lambda', 'mean')])

        scatters = []
        for assum in assumptions :
            ax = plt.scatter(
                x=results_dict[(assum, xaxis, 'mean')],
                y=results_dict[(assum, yaxis, 'mean')],
                c=harm_metrics_lamda_freq.loc[assum]['run'].values/config.N_RUNS,
                marker=MARKERS_MODELS[model_name],
                s=200,
                zorder=i,
                cmap=cmap,
                vmin=0, 
                vmax=1.,
                label=model_name
            )
            scatters.append(ax)

            y_low, y_up = confidence95(results_dict[(assum, yaxis, 'mean')], results_dict[(assum, yaxis, 'se')])
            x_low, x_up = confidence95(results_dict[(assum, xaxis, 'mean')], results_dict[(assum, xaxis, 'se')])
            plt.fill_betweenx(
                y=results_dict[(assum, yaxis, 'mean')], 
                x1=x_low,
                x2=x_up,
                color='darkgrey',
                alpha=.2, 
                zorder=i + 6)
            plt.fill_between(
                x=results_dict[(assum, xaxis, 'mean')], 
                y1=y_low,
                y2=y_up, 
                alpha=.2, 
                color='darkgrey', 
                zorder=i + 7)

            for axis in [xaxis, yaxis]:
                min_cur = min(results_dict[((assum, axis, 'mean'))])
                min_val[axis] = min(min_cur, min_val[axis])
            
                max_cur = max(results_dict[((assum, axis, 'mean'))])
                max_val[axis] = max(max_cur, max_val[axis])
            # Print also predicted
            if pred_sets:
                pred_sets = False

    if show_alpha:
        plt.axvline(alpha,  ls='--', color='black', zorder=35)
        plt.text(alpha, max_val[metric] + 0.025, f"{alpha}", ha='center', va='top', color='black', rotation=0, fontsize=48)
    
    plt.ylabel(AX_LABELS[yaxis])
    if assum == 'cI':
        plt.xlabel(AX_LABELS[xaxis]+' Bound')
    else:
        plt.xlabel(AX_LABELS[xaxis])
    
    ax = plt.gca()
   
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if save:
        path = f"{config.ROOT_DIR}/{config.plot_path}/{config.noise_level}/harm_vs_{metric}/"
        create_path(path)
        psets = "_PS" if pred_sets or 'Real' in models else ""
        plt.savefig(f"{path}/{len(models)}class_{assumptions[0]}_alpha{alpha}_split{config.calibration_split}{psets}.pdf", bbox_inches='tight')
        plt.clf()
    else:
        plt.show()

def plot_colorbar():
    """Saves the colobar in a separate file"""
    # Color bar size for stacked plots
    # fig, ax = plt.subplots(figsize=(1, 17.5))
    # Color bar for single plots 
    fig, ax = plt.subplots(figsize=(1, 12))
    cmap = mpl.colormaps['Blues']
    norm = mpl.colors.Normalize(0, 1)  # or vmin, vmax
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm, cmap), ax, pad=2)
    cbar.ax.tick_params(labelsize=48)
    plt.savefig(f"{config.ROOT_DIR}/{config.plot_path}/colorbar_small.pdf", bbox_inches='tight')

def plot_legend(models, pred_sets, assumptions, metric='accuracy'):
    """Saves the legend for real vs predicted"""
    handles = [
        plt.plot([],color='darkgray',ls="", marker=mark, zorder=60)[0] 
        for mark in [MARKERS_MODELS[model] for model in models]
    ]
    labels = [l for l in  models]
  
    fig = plt.legend(
        handles, 
        labels,
        columnspacing=1., 
        handlelength=0.01, 
        handleheight=0.4,
        markerscale=3.5,
        frameon=False,
        ncols=len(models),
        loc=LOCATION[assumptions[0]]
        )
    plt.gca().set_axis_off()
    plt.gcf().set_figheight(2)
    plt.gcf().set_figwidth(16)
    path = f"{config.ROOT_DIR}/{config.plot_path}/{config.noise_level}/harm_vs_{metric}/"
    create_path(path)
    psets = "_PS" if pred_sets or 'Real' in models else ""
    plt.savefig(f"{path}/legend_{len(models)}class_{assumptions[0]}_split{config.calibration_split}{psets}.pdf", bbox_inches='tight')
    plt.clf()
    plt.gcf().set_figheight(10)
    plt.gcf().set_figwidth(16)

def subplots_harm_vs_acc(
                   metric='accuracy',
                   xaxis='accuracy', 
                   yaxis='harm', 
                   models=['DensNet161'],
                   pred_sets=False, 
                   assumptions=['CF', 'cI'], 
                   show_alpha=True,
                   alpha=.1,
                   noise_level=110,
                   save=True
                   ):
    """Plots and saves the stacked plots of average counterfactual harm against the average accuracy
    for a fixed alpha and noise level"""
    def confidence95(mean, se):
        low = mean - 1.96*se
        up = mean + 1.96*se
        assert all(1.96*se < 0.01)
        return low, up
    
    def alpha_line_fn(xaxis, ax):
        if xaxis=='harm':
            return ax.axvline
        else:
            return ax.axhline

    cmap = plt.colormaps['Blues']
    min_val = defaultdict(lambda:1.)
    max_val = defaultdict(lambda:0.)
    
   
    fig, ax = plt.subplots(nrows=len(models), ncols=1, sharex=True)

    for i, model_name in enumerate(models):
    # set metric to 'metrics' for when pred_sets
        metr = 'metrics' if pred_sets else 'metric_2'
        harm_metrics_lamda_freq = metric_vs_harm(
            pred_sets=pred_sets, 
            metric=metr, 
            model_name=model_name,
            noise_level=noise_level,
            alpha=alpha,
        )
        
        results_dict = {}
        for assum in assumptions:
            for m in ['harm', metric]:
                for val in ['mean', 'se']:
                    results_dict[(assum, m, val)] = harm_metrics_lamda_freq.loc[assum][f"{m}_{val}"].values
        
        if metric == 'harm':
            for assum in assumptions:
                results_dict[(assum, 'lambda', 'mean')] = harm_metrics_lamda_freq.loc[assum].index.to_numpy()
                results_dict[(assum, 'lambda', 'se')] = np.zeros_like(results_dict[(assum, 'lambda', 'mean')])

        for assum in assumptions :
            sc = ax[i].scatter(
                x=results_dict[(assum, xaxis, 'mean')],
                y=results_dict[(assum, yaxis, 'mean')],
                c=harm_metrics_lamda_freq.loc[assum]['run'].values/config.N_RUNS,
                marker='o',
                s=200,
                zorder=i,
                cmap=cmap,
                vmin=0, 
                vmax=1.,
                label=model_name
            )

            y_low, y_up = confidence95(results_dict[(assum, yaxis, 'mean')], results_dict[(assum, yaxis, 'se')])
            x_low, x_up = confidence95(results_dict[(assum, xaxis, 'mean')], results_dict[(assum, xaxis, 'se')])
            ax[i].fill_betweenx(
                y=results_dict[(assum, yaxis, 'mean')], 
                x1=x_low,
                x2=x_up,
                color='darkgrey',
                alpha=.1, 
                zorder=i + 6)
            ax[i].fill_between(
                x=results_dict[(assum, xaxis, 'mean')], 
                y1=y_low,
                y2=y_up, 
                alpha=.1, 
                color='darkgrey', 
                zorder=i + 7)

            for axis in [xaxis, yaxis]:
                min_cur = min(results_dict[((assum, axis, 'mean'))])
                min_val[axis, i] = min(min_cur, min_val[axis])
            
                max_cur = max(results_dict[((assum, axis, 'mean'))])
                max_val[axis, i] = max(max_cur, max_val[axis])
            
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['top'].set_visible(False)

            leg = ax[i].legend(
                frameon=False,
                loc='upper right'
                )
            leg.legend_handles[0].set_visible(False)
           
    
            if show_alpha:
                alpha_plotter = alpha_line_fn(xaxis=xaxis, ax=ax[i])
                if i < 4:
                    ymin = -0.5
                else:
                    ymin = 0
                alpha_plotter(
                    alpha, 
                    ymin=ymin,
                    ls='--', 
                    color='black', 
                    zorder=35,
                    clip_on=False)
                if i == 0:
                    # To fix the height of annotation change the + 0.04 to another value. 
                    ax[i].text(alpha, max_val[metric, i] + 0.04, f"{alpha}", ha='center', va='top', color='black', rotation=0, fontsize=48)
    
    
    fig.supylabel(AX_LABELS[yaxis],x=-0.02)
    fig.supxlabel(AX_LABELS[xaxis], y=-0.01)
    if assum=='cI':
        fig.supxlabel(AX_LABELS[xaxis]+r' Bound', y=-0.01)

    if save:
        path = f"{config.ROOT_DIR}/{config.plot_path}/{noise_level}/harm_vs_{metric}/"
        create_path(path)
        psets = "_PS" if pred_sets or 'Real' in models else ""
        plt.savefig(f"{path}/all_{assumptions[0]}_alpha{alpha}_split{config.calibration_split}{psets}.pdf", bbox_inches='tight')
        plt.clf()
    else:
        plt.show()

  
if __name__=="__main__":
    """Produces all plots showing the average accuracy against the 
    average counterfactual harm bound for different alpha values
    for reach noise level, each model and each monotonicity 
    assumption (CF-->counterfactual, cI --> interventional)"""
    plot_colorbar()
    models = [
        r'VGG19',
        r'DenseNet161',
        r'GoogleNet',
        r'ResNet152',
        r'AlexNet',
    ]
    for noise_level in [80, 95, 110]:
        for assum in ['CF', 'cI']:
            for alpha in MAP_ASSUM_ALPHA[assum][noise_level]:
                subplots_harm_vs_acc(
                    yaxis='accuracy', 
                    xaxis='harm', 
                    metric='accuracy',
                    models=models,
                    pred_sets=False, 
                    assumptions=[assum], 
                    show_alpha=True,
                    alpha=alpha, 
                    noise_level=noise_level,
                    save=True
                )
                if config.noise_level == 110 and assum == 'CF':
                    plot_legend(['Real', 'Predicted'], True, [assum])
                    harm_vs_metric_per_model(
                        yaxis='accuracy', 
                        xaxis='harm', 
                        metric='accuracy',
                        models=[r'VGG19', r'VGG19'],
                        pred_sets=True, 
                        assumptions=[assum], 
                        show_alpha=True,
                        show_legend=False,
                        alpha=alpha, 
                        save=True
                    )

            