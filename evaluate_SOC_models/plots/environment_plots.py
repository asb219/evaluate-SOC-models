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
    'plot_data_vs_clay',
    'plot_data_vs_temperature',
    'plot_sampling_date_vs_clay',
    'plot_sampling_date_vs_temperature'
]


@_save_show_close
def plot_data_vs_clay(environment=None, *args, **kwargs):
    if environment is None:
        environment = pd.Series({
            profile: ForcingData(*profile).constant['clay']
            for profile in get_all_profiles()
        })
    fig, axes = plot_data_vs_environment(
        environment, *args, xlabel='clay content',
        show=False, close=False, **kwargs
    )
    xticks = list(range(0, 55, 10))
    for ax in axes.flatten():
        ax.set_xticks(xticks, [f'{k:.0f}%' for k in xticks])
        ax.set_xlim((0,55))
    return fig, axes


@_save_show_close
def plot_data_vs_temperature(environment=None, *args, **kwargs):
    if environment is None:
        environment = pd.Series({
            profile: ForcingData(*profile).dynamic.loc[
                '1970':'2009', 'Tsoil'].mean() - 273.15
            for profile in get_all_profiles()
        })
    fig, axes = plot_data_vs_environment(
        environment, *args, xlabel='temperature (°C)',
        show=False, close=False, **kwargs
    )
    for ax in axes.flatten():
        ax.set_xlim((5,32))
    return fig, axes


@_save_show_close
def plot_data_vs_environment(environment, plot_type='predicted',
        predicted=None, observed=None, error=None, *, plot_soc_logscale=None,
        normalized_to_2000=False, xlabel='', figsize=(10,6)):

    if plot_type == 'predicted':
        if plot_soc_logscale is None:
            plot_soc_logscale = True
        if observed is None:
            observed = AllObservedData().data
        if predicted is None:
            predicted = get_all_results()[0]
        data = {'observations': observed} | predicted

    elif plot_type in ('error', 'absolute_error'):
        if plot_soc_logscale is None:
            plot_soc_logscale = False
        if error is None:
            error = get_all_results()[1]
        if plot_type == 'absolute_error':
            data = {key: value.abs() for key, value in error.items()}
        else:
            data = error

    else:
        raise ValueError("`plot_type` must be in ('predicted', 'error',"
            f" 'absolute_error'), but '{plot_type}' provided")


    suffix = '_2000' if normalized_to_2000 else ''

    variables = [
        'soc', 'LF_c_perc', 'HF_c_perc',
        'bulk_14c'+suffix, 'LF_14c'+suffix, 'HF_14c'+suffix
    ]

    _plot_predicted = plot_type == 'predicted'
    _no_ylim = (0 if plot_type == 'absolute_error' else None, None)

    if plot_type == 'error':
        _title_prefix = 'Error in '
    elif plot_type == 'absolute_error':
        _title_prefix = 'Abs. error in '
    else:
        _title_prefix = ''

    options = {
        'soc': dict(
            title=_title_prefix + 'SOC stocks',
            ylabel='SOC stocks (kgC m$^{-2}$)',
            ylim=(None,None) if _plot_predicted else _no_ylim
        ),
        'LF_c_perc': dict(
            title=_title_prefix + 'POM contribution',
            ylabel=None,
            ylim=(0,100) if _plot_predicted else _no_ylim
        ),
        'HF_c_perc': dict(
            title=_title_prefix + 'MAOM contribution',
            ylabel=None,
            ylim=(0,100) if _plot_predicted else _no_ylim
        ),
        'bulk_14c'+suffix: dict(
            title=_title_prefix + 'Bulk SOC $\Delta^{14}$C',
            ylabel='$\Delta^{14}$C (‰)',
            ylim=(-150,250) if _plot_predicted else _no_ylim
        ),
        'LF_14c'+suffix: dict(
            title=_title_prefix + 'POM $\Delta^{14}$C',
            ylabel=None,
            ylim=(-150,250) if _plot_predicted else _no_ylim
        ),
        'HF_14c'+suffix: dict(
            title=_title_prefix + 'MAOM $\Delta^{14}$C',
            ylabel=None,
            ylim=(-150,250) if _plot_predicted else _no_ylim
        )
    }

    def scatter_plot(ax, x, y, *, marker='o', **kwargs):
        ax.plot(x, y, ls='', marker=marker, **kwargs)
        return ax

    def linregress_plot(ax, x, y, logscale=False, **kwargs):
        if logscale:
            result = scipy.stats.linregress(x, np.log(y))
            xy1 = (0, np.exp(result.intercept))
            xy2 = (1, np.exp(result.intercept + result.slope))
            ax.axline(xy1, xy2, **kwargs)
        else:
            result = scipy.stats.linregress(x, y)
            xy1 = (0, result.intercept)
            slope = result.slope
            ax.axline(xy1, slope=slope, **kwargs)
        return ax


    fig, axes = plt.subplots(
        nrows=2, ncols=3, sharey=False, sharex=True, figsize=figsize
    )

    subplot_labels = iter('('+chr(i)+') ' for i in range(ord('a'), ord('z')))

    for ax, variable, splabel in zip(axes.flatten(), variables, subplot_labels):
        for i, model_name in enumerate(['observations'] + SORTED_MODEL_NAMES):
            if model_name not in data:
                continue
            y = data[model_name][variable].dropna()
            if variable == 'soc':
                y = y * 10 # gC/cm2 -> kgC/m2
            x = environment[y.index]
            #corr = scipy.stats.pearsonr(x,y)
            #print(f'{variable}: {model_name}: r={corr.statistic:.3f}, p={corr.pvalue:.3f}')
            ax = scatter_plot(ax, x, y, color=f'C{i}', alpha=0.2, zorder=i+1)
            ax = linregress_plot(
                ax, x, y, color=f'C{i}', zorder=i+101, label=model_name,
                logscale=plot_soc_logscale and variable=='soc',
                lw=4 if model_name=='observations' else 2
            )

        ax.set_title(splabel + options[variable]['title'])
        ax.set_ylabel(options[variable]['ylabel'], ''), size=11)
        if plot_soc_logscale and variable == 'soc':
            ax.set_yscale('log')
        else:
            ax.set_ylim(options[variable]['ylim'])
        if 'perc' in variable:
            ax.set_yticks(ticks:=ax.get_yticks(), [f'{k:.0f}%' for k in ticks])

    axes[1,0].set_xlabel(xlabel, size=11)
    axes[1,1].set_xlabel(xlabel, size=11)
    axes[1,2].set_xlabel(xlabel, size=11)

    fig.legend(
        *axes[0,0].get_legend_handles_labels(), ncols=1,
        loc='center', bbox_to_anchor=(1.1, 0.5)
    )

    return fig, axes



@_save_show_close
def plot_sampling_date_vs_clay(environment=None, *args, **kwargs):
    if environment is None:
        environment = pd.Series({
            profile: ForcingData(*profile).constant['clay']
            for profile in get_all_profiles()
        })
    fig, ax = plot_sampling_date_vs_environment(
        environment, *args, xlabel='clay content',
        show=False, close=False, **kwargs
    )
    ax.set_xticks(ticks:=ax.get_xticks(), [f'{k:.0f}%' for k in ticks])
    ax.set_xlim((0,55))
    return fig, ax


@_save_show_close
def plot_sampling_date_vs_temperature(environment=None, *args, **kwargs):
    if environment is None:
        environment = pd.Series({
            profile: ForcingData(*profile).dynamic.loc[
                '1970':'2009', 'Tsoil'].mean() - 273.15
            for profile in get_all_profiles()
        })
    fig, ax = plot_sampling_date_vs_environment(
        environment, *args, xlabel='temperature (°C)',
        show=False, close=False, **kwargs
    )
    ax.set_xlim((5,32))
    return fig, ax


@_save_show_close
def plot_sampling_date_vs_environment(environment, observed=None, *,
        xlabel='', figsize=(4,3)):

    if observed is None:
        observed = AllObservedData().data

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    y = observed['date']
    x = environment[y.index]

    ax.plot(x, y, marker='o', ls='', alpha=1)
    ax.set_ylim((pd.to_datetime('1994'), None))
    ax.set_ylabel('sampling year')
    ax.set_xlabel(xlabel)

    return fig, ax
