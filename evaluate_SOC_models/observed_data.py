import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker

from data_manager import Data

from evaluate_SOC_models.data_sources import \
    ISRaDData, Graven2017CompiledRecordsData, CESM2LEOutputFileGroup

from evaluate_SOC_models.path import TOPSOIL_MIN_DEPTH, TOPSOIL_MAX_DEPTH
from evaluate_SOC_models.path import SAVEOUTPUTPATH, SAVEALLDATAPATH
from evaluate_SOC_models.utils import _save_show_close


__all__ = ['SelectedISRaDData', 'AllObservedData', 'ObservedData']


class SelectedISRaDData(Data):

    datasets = ['data']

    def __init__(self, *, save_pkl=False, save_csv=False, save_xlsx=False):

        savedir = SAVEALLDATAPATH / 'data'
        name = 'selected_israd'
        description = ('Observed data of 14C and relative C'
            ' of density fractions, and bulk 14C and SOC from ISRaD.')

        super().__init__(savedir, name, description,
            save_pkl=save_pkl, save_csv=save_csv, save_xlsx=save_xlsx)


    def _process_data(self):
        israd = ISRaDData(TOPSOIL_MIN_DEPTH, TOPSOIL_MAX_DEPTH)
        #israd.purge_savedir('*topsoil_data.pkl.gz', ask=False, well_behaved=False)

        df = israd['topsoil_data'].copy()

        # Drop profiles where there is no 14C data for the fractions
        df = df.dropna(how='all', subset=['HF_14c', 'fLF_14c', 'oLF_14c'])
        #df = df.dropna(how='all', subset=['HF_c_perc', 'fLF_c_perc', 'oLF_c_perc'])

        # Fill missing bulk 14C from integrated fraction data
        _14c = df[['HF_14c', 'fLF_14c', 'oLF_14c']].values
        _c_perc = df[['HF_c_perc', 'fLF_c_perc', 'oLF_c_perc']].values
        fill_bulk_14c = (_14c * _c_perc).sum(axis=1) / _c_perc.sum(axis=1)
        fill_bulk_14c = pd.Series(fill_bulk_14c, index=df.index)
        df['bulk_14c'] = df['bulk_14c'].fillna(fill_bulk_14c)

        # Drop profiles where there is no bulk 14C data
        df = df.dropna(subset='bulk_14c')

        # Compute LF data from fLF and oLF data
        df['LF_c_perc'] = df['fLF_c_perc'] + df['oLF_c_perc']
        df['LF_14c'] = (
            df['fLF_14c'] * df['fLF_c_perc'] + df['oLF_14c'] * df['oLF_c_perc']
        ) / df['LF_c_perc']

        # Add information from pro_info, site_info, entry_info
        pro_info = israd['pro_info'][[
            'pro_peatland',
            'pro_permafrost',
            'pro_thermokarst',
            'pro_usda_soil_order',
            'pro_soil_series',
            'pro_soil_taxon',
            'pro_land_cover',
            'pro_lc_phenology',
            'pro_lc_leaf_type',
        ]]
        df = df.join(pro_info)

        site_info = israd['site_info'][['site_lat', 'site_long', 'site_elevation']]
        df = df.join(site_info)

        entry_info = israd['entry_info'][[
            'doi', 'compilation_doi', 'bibliographical_reference'
        ]]
        df = df.join(entry_info)

        # # Get rid of permafrost, peatland, thermokarst, wetland soils
        # df = df[
        #     (df['pro_peatland'] != 'yes') &
        #     (df['pro_permafrost'] != 'yes') &
        #     (df['pro_thermokarst'] != 'yes') &
        #     (df['pro_land_cover'] != 'wetland')
        # ]
        #
        # # Select data from after 1990
        # df = df[df['date'] > '1990']

        return df.copy()


    @_save_show_close
    def plot_map(self, projection='Miller', plot_MAT=True, plot_clay=True):

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
        #ax.add_feature(cfeature.BORDERS.with_scale('110m'))
        ax.coastlines(zorder=102)

        data = self['data'].drop_duplicates(subset=['site_lat', 'site_long'])
        lat = data['site_lat'].values
        lon = data['site_long'].values
        if plot_clay:
            from evaluate_SOC_models.forcing_data import AllConstantForcingData
            color = AllConstantForcingData()['data']['clay'][data.index].values
            p = ax.scatter(lon, lat, transform=crs, zorder=200,
                alpha=1,
                marker='o',
                s=40, # marker size
                c=color, # marker face color
                edgecolors='black',
                linewidths=0.7 # marker edge width
            )
            fig.colorbar(p, label='Topsoil clay content (%)', boundaries=[0,10,20,30,40,50], extend='max')
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
                levels=[-20,-10,0,10,20,30,40], zorder=10) #, extend='min')
            fig.colorbar(p, label='Topsoil temperature (°C)') #, boundaries=[-10,0,10,20,30,40], extend='min')

        #if projection == 'LambertConformal':
        gl = ax.gridlines(draw_labels=True, lw=0.2, color='k', alpha=1, ls='-', zorder=150)
        gl.top_labels = False
        gl.right_labels = False
        if projection == 'Miller':
            gl.ylocator = mticker.FixedLocator([0, 20, 40,60])

        #plt.axis('off')
        #embed()

        return fig, ax


    @_save_show_close
    def plot_timeseries(self, figsize=(4,3.5)):
        markersize = 8 #5
        markers = {
            'bulk_14c': dict(
                ls='',
                color='C0',
                marker='o',
                markersize=markersize,
                alpha=0.4,
                # marker='o',
                # markersize=5,
                # #color='tab:blue',
                # markerfacecolor='mediumblue',
                # markeredgecolor='mediumblue',
                zorder=50,
                label='bulk SOC'
            ),
            'LF_14c': dict(
                ls='',
                color='C2',
                #marker='^',
                markersize=markersize+1,
                alpha=0.4,
                marker=8,
                # markersize=6,
                # #color='tab:orange',
                # markerfacecolor='none',
                # markeredgecolor='darkred',
                zorder=40,
                label='POM'
            ),
            'HF_14c': dict(
                ls='',
                color='C1',
                #marker='v',
                markersize=markersize+1,
                alpha=0.4,
                marker=9,
                # markersize=6,
                # #color='darkcyan',
                # markerfacecolor='none',
                # markeredgecolor='darkcyan',
                zorder=30,
                label='MAOM'
            )
        }
        x_offset = {'bulk_14c': 0, 'LF_14c': -0.3, 'HF_14c': +0.3}

        #Delta14C_atm = Graven2017CompiledRecordsData().Delta14C['NH']
        atmosphere = Graven2017CompiledRecordsData().Delta14C.loc['1900':, 'NH']
        data = self['data'].reset_index().set_index('date')

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


    @_save_show_close
    def plot_boxplot(self):
        fig = plt.figure(figsize=(4,4))
        data = self['data'][['bulk_14c', 'LF_14c', 'HF_14c']]
        labels = ['total SOC', 'light\nfraction', 'heavy\nfraction']
        plt.axhline(c='k', alpha=0.6, lw=1)
        plt.boxplot(data.values, vert=True, showmeans=True, labels=labels)
        plt.ylabel('$\Delta^{14}$C (‰)')
        return fig, None



class AllObservedData(Data):

    datasets = ['data']

    def __init__(self, *, save_pkl=False, save_csv=False, save_xlsx=False):

        savedir = SAVEALLDATAPATH / 'data'
        name = 'all_observed'
        description = ('All selected observed data of 14C and relative C'
            ' of density fractions, and bulk 14C and SOC from ISRaD.')

        super().__init__(savedir, name, description,
            save_pkl=save_pkl, save_csv=save_csv, save_xlsx=save_xlsx)

    def _process_data(self):
        obs = SelectedISRaDData().data[
            ['date', 'soc', 'bulk_14c'] +
            [f+v for v in ('_14c','_c_perc') for f in ('HF','LF','fLF','oLF')]
        ].copy()
        return obs



class ObservedData(Data):

    datasets = ['data']

    def __init__(self, entry_name, site_name, pro_name, *,
            save_pkl=False, save_csv=False, save_xlsx=False):

        savedir = SAVEOUTPUTPATH / entry_name / site_name / pro_name / 'data'
        name = 'observed'
        description = ('Observed data of 14C and relative C'
            ' of density fractions, and bulk 14C and SOC from ISRaD.'
            f' Data for {entry_name} / {site_name} / {pro_name}.')

        super().__init__(savedir, name, description,
            save_pkl=save_pkl, save_csv=save_csv, save_xlsx=save_xlsx)

        self.entry_name = entry_name
        self.site_name = site_name
        self.pro_name = pro_name


    def _process_data(self):
        """ Returns a pandas DataFrame with date as index """
        israd_index = (self.entry_name, self.site_name, self.pro_name)
        obs = AllObservedData().data.loc[[israd_index]].set_index('date')
        return obs.copy()
