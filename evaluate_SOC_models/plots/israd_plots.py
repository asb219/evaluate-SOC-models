"""
Map of sampling sites and time series plot of selected 14C data from ISRaD.

Copyright (C) 2024  Alexander S. Brunmayr  <asb219@ic.ac.uk>

This file is part of the ``evaluate_SOC_models`` python package, subject to
the GNU General Public License v3 (GPLv3). You should have received a copy
of GPLv3 along with this file. If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker

from evaluate_SOC_models.data.observed import SelectedISRaDData
from evaluate_SOC_models.data.forcing import AllConstantForcingData
from evaluate_SOC_models.data.sources import CESM2LEOutputFileGroup
from evaluate_SOC_models.data.sources import Graven2017CompiledRecordsData

from .utils import _save_show_close


__all__ = [
    'plot_israd_map',
    'plot_israd_timeseries'
]


@_save_show_close
def plot_israd_map(projection='Miller', plot_MAT=True, plot_clay=True):

    figwidth = 6
    figheight = 3
    if plot_MAT:
        figwidth += 2 # more space for colorbar on side
    if plot_clay:
        figwidth += 2 # more space for colorbar on side
    fig = plt.figure(figsize=(figwidth, figheight))

    if projection == 'Miller':
        projection_crs = ccrs.Miller()
        minlon,maxlon,minlat,maxlat = -168.5,41.5,-0.01,71.5
    elif projection == 'LambertConformal':
        projection_crs = ccrs.LambertConformal(central_longitude=-60)
        minlon,maxlon,minlat,maxlat = -168.5,40,8,71.5
    else:
        raise ValueError

    ax = fig.add_subplot(1, 1, 1, projection=projection_crs)

    crs = ccrs.PlateCarree()

    ax.set_extent([minlon,maxlon,minlat,maxlat], crs=crs)

    ax.add_feature(cfeature.LAND.with_scale('110m'), color='lightgrey', zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale('110m'), zorder=100)
    ax.add_feature(cfeature.LAKES.with_scale('110m'), zorder=101)
    ax.coastlines(zorder=102)

    data = SelectedISRaDData()['data'].drop_duplicates(
        subset=['site_lat', 'site_long']
    )
    lat = data['site_lat'].values
    lon = data['site_long'].values
    if plot_clay:
        color = AllConstantForcingData()['data']['clay'][data.index].values
        p = ax.scatter(lon, lat, transform=crs, zorder=200,
            alpha=1,
            marker='o',
            s=40, # marker size
            c=color//10*10+5, # binned marker face color
            edgecolors='black',
            linewidths=0.7 # marker edge width
        )
        fig.colorbar(p, label='Topsoil clay content (%)', extend='max',
            boundaries=[0,10,20,30,40,50,60], values=[5,15,25,35,45,55])
    else:
        color = 'red'
        ax.plot(lon, lat, ls='', transform=crs, zorder=200,
            alpha=1,
            marker='o',
            markersize=5,
            markerfacecolor=color,
            markeredgecolor='black',
            markeredgewidth=0.7
        )

    if plot_MAT:
        MAT_file = CESM2LEOutputFileGroup('TSOI', 'monthly')
        MAT_global = MAT_file.read()['TSOI'].sel(levgrnd=0.04) # 4cm depth
        MAT_global_mean = MAT_global.loc['1970':'2009'].mean(dim='time')
        lon = np.append(MAT_global_mean.lon.values, 360) # add longitude 360
        lat = MAT_global_mean.lat.values
        lons, lats = np.meshgrid(lon, lat)
        MAT_data = MAT_global_mean.values - 273.15
        # Add data at 360 degrees (equals data at 0 degrees)
        MAT_data = np.append(MAT_data, MAT_data[:,[0]], axis=1)
        p = ax.contourf(lons, lats, MAT_data, transform=crs, cmap='RdBu_r',
            levels=[-20,-10,0,10,20,30,40], zorder=10)
        fig.colorbar(p, label='Topsoil temperature (°C)')

    gl = ax.gridlines(draw_labels=True, lw=0.2, color='k', alpha=1, ls='-', zorder=150)
    gl.top_labels = False
    gl.right_labels = False
    if projection == 'Miller':
        gl.ylocator = mticker.FixedLocator([0, 20, 40,60])

    return fig, ax


@_save_show_close
def plot_israd_timeseries(figsize=(4,3.5)):
    markersize = 8 #5
    markers = {
        'bulk_14c': dict(
            ls='',
            color='C0',
            marker='o',
            markersize=markersize,
            alpha=0.4,
            markeredgecolor='none',
            zorder=50,
            label='bulk SOC'
        ),
        'LF_14c': dict(
            ls='',
            color='C2',
            markersize=markersize+1,
            alpha=0.4,
            marker=8,
            markeredgecolor='none',
            zorder=40,
            label='POM'
        ),
        'HF_14c': dict(
            ls='',
            color='C1',
            markersize=markersize+1,
            alpha=0.4,
            marker=9,
            markeredgecolor='none',
            zorder=30,
            label='MAOM'
        )
    }
    x_offset = {'bulk_14c': 0, 'LF_14c': -0.3, 'HF_14c': +0.3}

    atmosphere = Graven2017CompiledRecordsData().Delta14C.loc['1900':, 'NH']
    data = SelectedISRaDData()['data'].reset_index().set_index('date')

    fig = plt.figure(figsize=figsize)
    plt.axhline(y=0, c='k', alpha=0.6, lw=1)
    plt.plot(atmosphere.index.year, atmosphere, c='k', lw=3, label='atmospheric CO$_2$')
    for var in ['bulk_14c', 'LF_14c', 'HF_14c']:
        plt.plot(data.index.year+x_offset[var], data[var], **markers[var])
    plt.xlim((1945, 2020))
    #plt.ylim((-200, None))
    plt.legend(loc='upper right')
    plt.xlabel('year', size=12)
    plt.ylabel('$\Delta^{14}$C (‰)', size=12)

    return fig, None
