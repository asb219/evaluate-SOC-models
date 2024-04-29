import numpy as np
import pandas as pd
import scipy.optimize
#import scipy.stats
import matplotlib.pyplot as plt

from evaluate_SOC_models.observed_data import AllObservedData
from evaluate_SOC_models.forcing_data import ForcingData
from evaluate_SOC_models.results import \
    SORTED_MODEL_NAMES, get_all_profiles, get_all_results

from .utils import _save_show_close


__all__ = [
    'plot_predicted_vs_clay',
    'plot_predicted_vs_temperature'
]


@_save_show_close
def plot_predicted_vs_clay(environment=None, *args, **kwargs):
    if environment is None:
        environment = pd.Series({
            profile: ForcingData(*profile).constant['clay']
            for profile in get_all_profiles()
        })
    fig, axes = plot_predicted_vs_environment(
        environment, *args, xlabel='clay content', xlim=(0,55),
        show=False, close=False, **kwargs
    )
    for ax in axes.flatten():
        ax.set_xticks(ticks:=ax.get_xticks(), [f'{k:.0f}%' for k in ticks])
        ax.set_xlim((0,55))
    return fig, axes


@_save_show_close
def plot_predicted_vs_temperature(environment=None, *args, **kwargs):
    if environment is None:
        environment = pd.Series({
            profile: ForcingData(*profile).dynamic.loc[
                '1970':'2009', 'Tsoil'].mean() - 273.15
            for profile in get_all_profiles()
        })
    fig, axes = plot_predicted_vs_environment(
        environment, *args, xlabel='temperature (°C)', xlim=(5,35),
        show=False, close=False, **kwargs
    )
    for ax in axes.flatten():
        ax.set_xlim((5,32))
    return fig, axes


@_save_show_close
def plot_predicted_vs_environment(environment, predicted=None, observed=None, *,
        normalized_to_2000=False, xlabel='', xlim=(None,None), figsize=(10,6)):

    if observed is None:
        observed = AllObservedData().data
    if predicted is None:
        predicted = get_all_results()[0]

    suffix = '_2000' if normalized_to_2000 else ''

    options = {
        'soc': dict(
            title='Total SOC stocks',
            ylabel='SOC stocks (kgC m$^{-2}$)',
            ylim=(None,None)
        ),
        'LF_c_perc': dict(
            title='POM contribution',
            ylabel=None,
            ylim=(0,100)
        ),
        'HF_c_perc': dict(
            title='MAOM contribution',
            ylabel=None,
            ylim=(0,100)
        ),
        'bulk_14c'+suffix: dict(
            title='Bulk SOC $\Delta^{14}$C',
            ylabel='$\Delta^{14}$C (‰)',
            ylim=(-150,250)
        ),
        'LF_14c'+suffix: dict(
            title='POM $\Delta^{14}$C',
            ylabel=None,
            ylim=(-150,250)
        ),
        'HF_14c'+suffix: dict(
            title='MAOM $\Delta^{14}$C',
            ylabel=None,
            ylim=(-150,250)
        )
    }

    def scatter_plot(ax, x, y, color, zorder):
        ax.plot(x, y, color=color, marker='o', ls='', zorder=zorder, alpha=0.2)
        return ax

    def linregress_plot(ax, x, y, variable, label, color, lw, zorder):
        if variable == 'soc': # use logscale
            result = scipy.stats.linregress(x, np.log(y))
            xy1 = (0, np.exp(result.intercept))
            xy2 = (50, np.exp(result.intercept + result.slope*50))
            ax.axline(xy1, xy2, color=color, lw=lw, zorder=zorder, label=label)
        else:
            result = scipy.stats.linregress(x, y)
            xy1 = (0, result.intercept)
            slope = result.slope
            ax.axline(xy1, slope=slope, color=color, lw=lw, zorder=zorder)
        return ax


    variables = [
        'soc', 'LF_c_perc', 'HF_c_perc',
        'bulk_14c'+suffix, 'LF_14c'+suffix, 'HF_14c'+suffix
    ]

    fig, axes = plt.subplots(
        nrows=2, ncols=3, sharey=False, sharex=True, figsize=figsize
    )

    subplot_labels = iter('('+chr(i)+') ' for i in range(ord('a'), ord('z')))

    for ax, variable, splabel in zip(axes.flatten(), variables, subplot_labels):

        # Observed data
        y = observed[variable].dropna()
        if variable == 'soc':
            y = y * 10 # gC/cm2 -> kgC/m2
        x = environment[y.index]
        #corr = scipy.stats.pearsonr(x,y)
        #print(f'{variable}: observed: r={corr.statistic:.3f}, p={corr.pvalue:.3f}')
        label = 'observations'
        color = 'C0'
        ax = scatter_plot(ax, x, y, color, zorder=0)
        ax = linregress_plot(ax, x, y, variable, label, color, lw=4, zorder=100)

        # Predicted data
        for i, model_name in enumerate(SORTED_MODEL_NAMES):
            if model_name not in predicted:
                continue
            y = predicted[model_name][variable].dropna()
            if variable == 'soc':
                y = y * 10 # gC/cm2 -> kgC/m2
            x = environment[y.index]
            #corr = scipy.stats.pearsonr(x,y)
            #print(f'{variable}: {model_name}: r={corr.statistic:.3f}, p={corr.pvalue:.3f}')
            label = model_name
            color = f'C{i+1}'
            ax = scatter_plot(ax, x, y, color, zorder=i+1)
            ax = linregress_plot(ax, x, y, variable, label, color, lw=2, zorder=i+101)

        ax.set_title(splabel + options[variable]['title'])
        ax.set_xlim(xlim)
        ax.set_ylabel(options[variable]['ylabel'], size=11)
        if variable != 'soc':
            ax.set_ylim(options[variable]['ylim'])
        if 'perc' in variable:
            ax.set_yticks(ticks:=ax.get_yticks(), [f'{k:.0f}%' for k in ticks])
        if variable == 'soc':
            ax.set_yscale('log')

    axes[1,0].set_xlabel(xlabel, size=11)
    axes[1,1].set_xlabel(xlabel, size=11)
    axes[1,2].set_xlabel(xlabel, size=11)

    fig.legend(
        *axes[0,0].get_legend_handles_labels(), ncols=1,
        loc='center', bbox_to_anchor=(1.1, 0.5)
    )
    fig.tight_layout()

    return fig, axes
