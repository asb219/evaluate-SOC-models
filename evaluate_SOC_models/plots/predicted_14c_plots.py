"""
Plots of predicted and observed 14C vs time.

Copyright (C) 2024  Alexander S. Brunmayr  <asb219@ic.ac.uk>

This file is part of the ``evaluate_SOC_models`` python package, subject to
the GNU General Public License v3 (GPLv3). You should have received a copy
of GPLv3 along with this file. If not, see <https://www.gnu.org/licenses/>.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from evaluate_SOC_models.data.observed import SelectedISRaDData
from evaluate_SOC_models.data.sources import Graven2017CompiledRecordsData
from evaluate_SOC_models.models import \
    MENDData, MillennialData, SOMicData, CORPSEData, MIMICSData

from .utils import _save_show_close


__all__ = [
    'plot_predicted_14C',
    'plot_predicted_14C_all_models'
]


@_save_show_close
def plot_predicted_14C(model, profile, *,
        figsize=(6,5), ylim=(None, None), t0='1945', t1='2020'):

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    atmosphere = Graven2017CompiledRecordsData().Delta14C.NH

    t0, t1 = str(t0), str(t1)

    m = model(*profile)
    ax.axhline(y=0, color='k', alpha=0.7, zorder=-10, lw=0.8)
    ax.plot(atmosphere, c='k', lw=3, label='atmospheric CO$_2$', zorder=-1)
    ax.plot(m.output.loc[t0:t1, 'bulk_14c'], c='C0', label='predicted bulk SOC', zorder=5, ls='--', lw=2)
    ax.plot(m.output.loc[t0:t1, 'LF_14c'], c='C2', label='predicted POM', zorder=2, lw=2)
    ax.plot(m.output.loc[t0:t1, 'HF_14c'], c='C1', label='predicted MAOM', zorder=1, lw=2)
    ax.plot(m.observed['bulk_14c'], c='C0', marker='o', markersize=7, label='observed bulk SOC', ls='', zorder=15)
    ax.plot(m.observed['LF_14c'], c='C2', marker=8, markersize=10, label='observed POM', ls='', zorder=12)
    ax.plot(m.observed['HF_14c'], c='C1', marker=9, markersize=10, label='observed MAOM', ls='', zorder=11)
    ax.set_title(m.model_name + '\n' + ', '.join(profile), size=12)
    ax.set_xlim((pd.to_datetime(t0), pd.to_datetime(t1)))
    ax.set_ylim(ylim)

    ax.set_xlabel('year', size=12)
    ax.set_ylabel('$\Delta^{14}$C (‰)', size=12)

    plt.legend(loc='upper right')
    plt.tight_layout()

    return fig, ax


@_save_show_close
def plot_predicted_14C_all_models(profile, *,
        figsize=(10,5), ylim=(None, None), t0='1945', t1='2020'):

    fig, axes = plt.subplots(nrows=2, ncols=3, sharey=True, sharex=True, figsize=figsize)

    atmosphere = Graven2017CompiledRecordsData().Delta14C.NH

    sorted_models = (MENDData, MillennialData, SOMicData, CORPSEData, MIMICSData)
    sorted_axes = (axes[0,0], axes[0,1], axes[1,0], axes[1,1], axes[1,2])
    subplot_labels = iter('('+chr(i)+') ' for i in range(ord('a'), ord('z')))

    t0, t1 = str(t0), str(t1)

    for model, ax, splabel in zip(sorted_models, sorted_axes, subplot_labels):
        m = model(*profile)
        ax.axhline(y=0, color='k', alpha=0.7, zorder=-10, lw=0.8)
        ax.plot(atmosphere, c='k', lw=3, label='atmospheric CO$_2$', zorder=-1)
        ax.plot(m.output.loc[t0:t1, 'bulk_14c'], c='C0', label='predicted bulk SOC', zorder=5, ls='--', lw=2)
        ax.plot(m.output.loc[t0:t1, 'LF_14c'], c='C2', label='predicted POM', zorder=2, lw=2)
        ax.plot(m.output.loc[t0:t1, 'HF_14c'], c='C1', label='predicted MAOM', zorder=1, lw=2)
        ax.plot(m.observed['bulk_14c'], c='C0', marker='o', markersize=7, label='observed bulk SOC', ls='', zorder=15)
        ax.plot(m.observed['LF_14c'], c='C2', marker=8, markersize=10, label='observed POM', ls='', zorder=12)
        ax.plot(m.observed['HF_14c'], c='C1', marker=9, markersize=10, label='observed MAOM', ls='', zorder=11)
        ax.set_title(splabel + m.model_name, x=0.6, y=0.83, size=12)
        ax.set_xlim((pd.to_datetime(t0), pd.to_datetime(t1)))
        ax.set_ylim(ylim)

    axes[1,0].set_xlabel('year', size=12)
    axes[1,1].set_xlabel('year', size=12)
    axes[1,2].set_xlabel('year', size=12)
    axes[0,0].set_ylabel('$\Delta^{14}$C (‰)', size=12)
    axes[1,0].set_ylabel('$\Delta^{14}$C (‰)', size=12)
    axes[0,2].set_axis_off() # remove axes to make space for legend
    fig.legend(*ax.get_legend_handles_labels(), loc='center', bbox_to_anchor=(0.84, 0.78), ncols=1)
    fig.tight_layout()

    return fig, axes
