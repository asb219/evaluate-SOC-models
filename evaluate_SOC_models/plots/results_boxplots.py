import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from evaluate_SOC_models.results import SORTED_MODEL_NAMES, get_all_results
from evaluate_SOC_models.observed_data import AllObservedData

from .utils import _save_show_close


__all__ = [
    'plot_boxplots_C',
    'plot_boxplots_14C'
]


def plot_boxplots_C(*args, **kwargs):
    return _plot_boxplots(['soc', 'LF_c_perc', 'HF_c_perc'], *args, **kwargs)


def plot_boxplots_14C(*args, **kwargs):
    return _plot_boxplots(['bulk_14c', 'LF_14c', 'HF_14c'], *args, **kwargs)


@_save_show_close
def _plot_boxplots(variables, predicted=None, observed=None):

    if observed is None:
        observed = AllObservedData().data
    if predicted is None:
        predicted = get_all_results()[0]

    order = ['observed'] + SORTED_MODEL_NAMES
    palette = {n:'C'+str(i) for i,n in enumerate(order)}
    flierprops = dict(marker='d', color='k', markersize=3, ls='none')
    title_dict = {
        'bulk_14c': 'Bulk SOC $\Delta^{14}$C',
        'LF_14c': 'POM $\Delta^{14}$C',
        'HF_14c': 'MAOM $\Delta^{14}$C',
        'soc': 'Total SOC stocks',
        'LF_c_perc': 'POM contribution',
        'HF_c_perc': 'MAOM contribution'
    }

    fig, axes = plt.subplots(nrows=1, ncols=len(variables), figsize=(10,4), sharex=True)

    obs = observed.set_index('date', append=True)

    subplot_labels = iter('('+chr(i)+') ' for i in range(ord('a'), ord('z')))

    for ax, variable, splabel in zip(axes, variables, subplot_labels):
        data = pd.DataFrame({
            model_name: model_predictions[variable]
            for model_name, model_predictions in predicted.items()
        })
        data['observed'] = obs[variable]

        if variable == 'soc':
            data = data * 10 # gC/cm2 -> kgC/m2

        ax = sns.boxplot(data=data, order=order, palette=palette,
            showfliers=True, flierprops=flierprops, ax=ax)

        if variable.endswith('14c'):
            ax.axhline(0, zorder=-10, color='k', alpha=0.5, lw=0.8)
            ax.set_ylim((-100, None))

        elif variable.endswith('perc'):
            ax.set_ylim((0,100))
            ax.set_yticks(ticks:=ax.get_yticks(), [f'{k:.0f}%' for k in ticks])

        elif variable == 'soc':
            ax.set_yscale('log')
            ax.set_ylabel('SOC stocks (kgC m$^{-2}$)', size=11)

        ax.set_title(splabel + title_dict[variable], size=13)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(),
            rotation=40, horizontalalignment='right', rotation_mode='anchor')

    if variables[0].endswith('14c'):
        axes[0].set_ylabel('$\Delta^{14}$C (‰)', size=11)

    fig.tight_layout()

    return fig, axes
