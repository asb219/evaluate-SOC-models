import numpy as np
import matplotlib.pyplot as plt

from evaluate_SOC_models.data.observed import AllObservedData
from evaluate_SOC_models.results import \
    SORTED_MODEL_NAMES, get_all_models, get_results_all_profiles, get_all_results

from .utils import _save_show_close


__all__ = [
    'plot_predicted_vs_observed',
    'plot_predicted_vs_observed_all_models'
]


@_save_show_close
def plot_predicted_vs_observed(model, predicted=None, observed=None, *,
        fig=None, axes=None, figsize=(11,7)):

    if predicted is None:
        predicted = get_results_all_profiles(model)[0]
    if observed is None:
        observed = AllObservedData().data

    title_dict = {
        'bulk_14c': 'Bulk SOC $\Delta^{14}$C',
        'LF_14c': 'POM $\Delta^{14}$C',
        'HF_14c': 'MAOM $\Delta^{14}$C',
        'soc': 'SOC stocks',
        'LF_c_perc': 'POM contribution',
        'HF_c_perc': 'MAOM contribution'
    }
    color = 'C{}'.format(SORTED_MODEL_NAMES.index(model.model_name) + 1)

    if axes is None:
        if fig is not None:
            raise ValueError
        fig, axes = plt.subplots(
            nrows=2, ncols=3, sharey=False, sharex=False, figsize=figsize
        )

    variables = ['soc', 'LF_c_perc', 'HF_c_perc', 'bulk_14c', 'LF_14c', 'HF_14c']

    subplot_labels = iter('('+chr(i)+') ' for i in range(ord('a'), ord('z')))

    for ax, variable, splabel in zip(axes.flatten(), variables, subplot_labels):

        y = predicted[variable] #.dropna()
        x = observed[variable].loc[y.index.droplevel(-1)]

        if variable == 'soc':
            y = y * 10 # gC/cm2 -> kgC/m2
            x = x * 10 # gC/cm2 -> kgC/m2

        ax.plot(x, y, color=color, marker='.', ls='')
        ax.axline((0,0), slope=1, color='black', zorder=-1)

        ax.set_title(splabel + title_dict[variable])

        if variable.endswith('c_perc'):
            ax.set_ylim((0,100))
            ax.set_xlim((0,100))
            ax.set_yticks(ticks:=ax.get_yticks(), [f'{k:.0f}%' for k in ticks])
            ax.set_xticks(ticks:=ax.get_xticks(), [f'{k:.0f}%' for k in ticks])
        else:
            maximum = max(np.nanmax(x), np.nanmax(y))
            if variable == 'soc':
                ax.set_ylim((0,maximum+0.1))
                ax.set_xlim((0,maximum+0.1))
            else:
                minimum = min(np.nanmin(x), np.nanmin(y))
                ax.set_ylim((minimum-20,maximum+20))
                ax.set_xlim((minimum-20,maximum+20))

    axes[1,0].set_xlabel('observed', size=12)
    axes[1,1].set_xlabel('observed', size=12)
    axes[1,2].set_xlabel('observed', size=12)
    axes[0,0].set_ylabel('predicted\nSOC stocks (kgC m$^{-2}$)', size=12)
    axes[1,0].set_ylabel('predicted\n$\Delta^{14}$C (‰)', size=12)

    fig.suptitle(model.model_name, size=14)

    return fig, axes


@_save_show_close
def plot_predicted_vs_observed_all_models(
        models=None, predicted=None, observed=None, *,
        fig=None, axes=None, **kwargs
    ):

    if models is None:
        models = get_all_models()
    if predicted is None:
        predicted = get_all_results(models=models)[0]
    if observed is None:
        observed = AllObservedData().data

    for model in models:
        fig, axes = plot_predicted_vs_observed(
            model, predicted[model.model_name], observed,
            fig=fig, axes=axes, show=False, close=False, **kwargs)

    fig.suptitle('')

    return fig, axes
