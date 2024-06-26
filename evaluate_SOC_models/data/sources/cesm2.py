"""
CESM2 Large Ensemble (CESM2-LE) output data.

Citation for CESM2-LE dataset:
    IBS Center for Climate Physics, et al. (2021). "CESM2 Large
    Ensemble Community Project". UCAR/NCAR Climate Data Gateway.
    https://doi.org/10.26024/KGMP-C556

Citation for CESM2:
    Danabasoglu, G., et al. (2020). "The Community Earth System Model
    Version 2 (CESM2)". Journal of Advances in Modeling Earth Systems, 12.
    https://doi.org/10.1029/2019MS001916


Copyright (C) 2024  Alexander S. Brunmayr  <asb219@ic.ac.uk>

This file is part of the ``evaluate_SOC_models`` python package, subject to
the GNU General Public License v3 (GPLv3). You should have received a copy
of GPLv3 along with this file. If not, see <https://www.gnu.org/licenses/>.
"""

from functools import reduce
import numpy as np
import pandas as pd
import xarray as xr

from data_manager.data import Data
from data_manager.file import \
    XarrayNetCDFFile, XarrayNetCDFFileGroup, FileFromURL, FileGroupFromDownload

from evaluate_SOC_models.path import DOWNLOADPATH, DATAPATH


__all__ = [
    'CESM2LEOutputFile',
    'CESM2LEOutputFileGroup',
    'CESM2LEOutputData'
]


class CESM2LEOutputFile(XarrayNetCDFFile, FileFromURL):
    """
    CESM2 Large Ensemble output,
    https://www.earthsystemgrid.org/dataset/ucar.cgd.cesm2le.output.html

    Project page: https://www.cesm.ucar.edu/projects/community-projects/LENS2/
    """

    # See https://www.cesm.ucar.edu/models/cesm2/naming_conventions.html
    # for case_id and output_id naming convention.

    _default_options = dict(
        code_base = 'e21', # CESM2.1
        compset_shortname = 'BHISTcmip6', # historical with CMIP6 forcing
        res_shortname = 'f09_g17', # resolution
        desc_string = 'LE2-1301', # description
        nnn = '001', # 3-digit identifier

        scomp = 'clm2', # Community Land Model 2
        otype = 'h0', # history 0
        variable_name = '{variable:s}', # e.g. NPP, TSOI, H2OSOI

        DL_link_base_template = (
            'https://tds.ucar.edu/thredds/fileServer/datazone/campaign/cgd/'
            'cesm/CESM2-LE/lnd/proc/tseries/{freq:s}/{variable:s}/'
        ) # freq = 'day_365' for annual average, 'month_1' for monthly average
    )

    def __init__(self, variable, dates, freq, **kwargs):

        options = {
            name: kwargs.get(name, value)
            for name,value in self._default_options.items()
        }

        compset_char = options['compset_shortname'][0].lower()

        case_id = '.'.join((
            compset_char, options['code_base'], options['compset_shortname'],
            options['res_shortname'], options['desc_string'], options['nnn']
        )) # e.g. 'b.e21.BHISTcmip6.f09_g17.LE2-1301.001'

        output_id = '.'.join((
            case_id, options['scomp'], options['otype'],
            options['variable_name'], dates
        )).format(variable=variable)
        # e.g. 'b.e21.BHISTcmip6.f09_g17.LE2-1301.001.clm2.h0.NPP.185001-185912'

        DL_link_base = options['DL_link_base_template'].format(
            variable=variable, freq=freq)

        filename = output_id + '.nc'
        DL_link = DL_link_base + filename
        savedir = DOWNLOADPATH/'CESM2_LE_output'

        super().__init__(savedir / filename, DL_link)

        self.case_id = case_id
        self.output_id = output_id
        self.dates = dates
        self.freq = freq



class CESM2LEOutputFileGroup(XarrayNetCDFFileGroup, FileGroupFromDownload):

    def __init__(self, variable, time_resolution, **kwargs):

        if time_resolution == 'monthly':
            freq = 'month_1'
            _dates = '{start:04d}01-{end:04d}12'
            _2010_2014 = '201001-201412'
        elif time_resolution == 'annual':
            freq = 'day_365'
            _dates = '{start:04d}0101-{end:04d}0101'
            _2010_2014 = '20100101-20141231'
        else:
            raise NotImplementedError

        files = pd.Series({
            year: CESM2LEOutputFile(
                variable, _dates.format(start=year, end=year+9), freq, **kwargs
            ) for year in range(1850,2000+1,10)
        }, dtype='O', name=f'CESM2-LE output files for {variable}')

        files[2010] = CESM2LEOutputFile(variable, _2010_2014, freq, **kwargs)

        files.index.name = 'start_year'

        super().__init__(files, name=f'CESM2-LE output for {variable}')



class CESM2LEOutputData(Data):
    " CESM2 Large Ensemble Output "
    " https://www.earthsystemgrid.org/dataset/ucar.cgd.cesm2le.output.html "


    datasets = [
        'GPP', 'NPP', 'NPP_bg', 'NPP_ag', 'Tsoil', 'Wsoil',
        'Nlit', 'Clit', 'delta13Clit', 'Delta14Clit',
        'Cveg', 'delta13Cveg', 'Delta14Cveg',
        'Csoil', 'Delta14Csoil',
        'C_litterfall', 'C_litterHR'
    ]

    _variables = [
        'GPP', 'NPP', 'BGNPP', 'AGNPP', 'TSOI', 'H2OSOI',
        'TOTLITN', 'TOTLITC', 'C13_TOTLITC', 'C14_TOTLITC',
        'TOTVEGC', 'C13_TOTVEGC', 'C14_TOTVEGC',
        'SOILC_vr', 'C14_SOILC_vr',
        'LITFALL', 'LITTERC_HR'
    ]


    def __init__(self, lat, lon, top=0, bot=20, *,
            save_pkl=True, save_csv=False, save_xlsx=False):

        if lon < 0:
            lon += 360 # convert negative degrees west to degrees east

        lat = np.round(float(lat),2)
        lon = np.round(float(lon),2)
        top = int(np.round(top,0))
        bot = int(np.round(bot,0))

        identifier = f'lat{lat:+06.2f}_lon{lon:06.2f}'
        savedir = DATAPATH / 'CESM2-LENS' / identifier
        name = 'cesm2le'
        description = (
            f'CESM2-LENS output data for lat,lon = {lat:.2f},{lon:.2f}, '
            f'averaged over the {top}-{bot}cm soil depth where appropriate.'
        )

        super().__init__(savedir, name, description,
            save_pkl=save_pkl, save_csv=save_csv, save_xlsx=save_xlsx)

        # Specify top and bottom boundaries of soil layer in file naming
        for d in ['Tsoil', 'Wsoil', 'Csoil', 'Delta14Csoil']:
            d_t_b = d + f'_top{top}_bot{bot}'
            name_d_t_b = name + '_' + d_t_b
            self._file_groups['pickle'][d].filename = name_d_t_b + '.pkl.gz'
            self._file_groups['csv'][d].filename = name_d_t_b + '.csv'
            self._file_groups['excel'][d].sheet_name = d_t_b

        self.lat = lat
        self.lon = lon
        self.top = top # cm, top of depth layer
        self.bot = bot # cm, bottom of depth layer

        self.sourcefile_groups = {variable:
            CESM2LEOutputFileGroup(variable, 'annual', otype='h3')
            if variable in ('SOILC_vr', 'C14_SOILC_vr') else
            CESM2LEOutputFileGroup(variable, 'monthly', otype='h0')
            for variable in self._variables
        }


    def _process_GPP(self):
        GPP = self._read_data('GPP')
        GPP.name = 'gross primary productivity (gC/m2/s)'
        return GPP


    def _process_NPP(self):
        NPP = self._read_data('NPP')
        NPP = NPP.where(NPP>0, 0) # get rid of negative NPP
        NPP.name = 'net primary productivity (gC/m2/s)'
        return NPP


    def _process_NPP_bg(self):
        NPP = self._read_data('BGNPP')
        NPP = NPP.where(NPP>0, 0) # get rid of negative NPP
        NPP.name = 'below-ground net primary productivity (gC/m2/s)'
        return NPP


    def _process_NPP_ag(self):
        NPP = self._read_data('AGNPP')
        NPP = NPP.where(NPP>0, 0) # get rid of negative NPP
        NPP.name = 'above-ground net primary productivity (gC/m2/s)'
        return NPP


    def _process_Tsoil(self):
        Tsoil = self._read_data('TSOI')
        Tsoil.name = 'soil temperature (K)'
        return Tsoil


    def _process_Wsoil(self):
        Wsoil = self._read_data('H2OSOI')
        Wsoil.name = 'soil moisture (mm3/mm3)'
        return Wsoil


    def _process_Nlit(self):
        Nlit = self._read_data('TOTLITN')
        Nlit.name = 'litter N (gN/m2)'
        return Nlit


    def _process_Clit(self):
        Clit = self._read_data('TOTLITC')
        Clit.name = 'litter C (gC/m2)'
        return Clit


    def _process_delta13Clit(self):
        C = self['Clit'] # gC/m2
        C13 = self._read_data('C13_TOTLITC') # gC13/m2
        delta13Clit = self._get_delta13C(C,C13)
        delta13Clit.name = 'delta13C of litter (permille)'
        return delta13Clit


    def _process_Delta14Clit(self):
        C = self['Clit'] # gC/m2
        C14 = self._read_data('C14_TOTLITC') # gC14/m2
        Delta14Clit = self._get_Delta14C(C,C14)
        Delta14Clit.name = 'Delta14C of litter (permille)'
        return Delta14Clit


    def _process_Cveg(self):
        Cveg = self._read_data('TOTVEGC')
        Cveg.name = 'vegetation C (gC/m2)'
        return Cveg


    def _process_delta13Cveg(self):
        C = self['Cveg']
        C13 = self._read_data('C13_TOTVEGC')
        delta13Cveg = self._get_delta13C(C,C13)
        delta13Cveg.name = 'delta13C of vegetation (permille)'
        return delta13Cveg


    def _process_Delta14Cveg(self):
        C = self['Cveg']
        C14 = self._read_data('C14_TOTVEGC')
        Delta14Cveg = self._get_Delta14C(C,C14)
        Delta14Cveg.name = 'Delta14C of vegetation (permille)'
        return Delta14Cveg


    def _process_Csoil(self):
        Csoil = self._read_data('SOILC_vr') # gC/m3
        Csoil *= (self.bot - self.top) / 100 # gC/m3 -> gC/m2
        Csoil.name = 'soil C (gC/m2)'
        return Csoil


    def _process_Delta14Csoil(self):
        C = self['Csoil'] # gC/m2
        C14 = self._read_data('C14_SOILC_vr') # gC14/m3
        C14 *= (self.bot - self.top) / 100 # gC14/m3 -> gC14/m2
        Delta14Csoil = self._get_Delta14C(C,C14)
        Delta14Csoil.name = 'Delta14C of soil (permille)'
        return Delta14Csoil


    def _process_C_litterfall(self):
        C_litterfall = self._read_data('LITFALL')
        C_litterfall.name = 'litterfall C flux (gC/m2/s)'
        return C_litterfall


    def _process_C_litterHR(self):
        C_HR_litter = self._read_data('LITTERC_HR')
        C_HR_litter.name = 'litter heterotrophic respiration C flux (gC/m2/s)'
        return C_HR_litter


    def _get_delta13C(self, C, C13):
        if False:
            C = C / 12.01 # convert gC to mol
            C13 = C13 / 13 # convert gC13 to mol
            C12 = C - C13
        else:
            C12 = C - C13
        standard13C = 0.0112372 # VPBD standard from IAEA
        delta13C = ( C13/C12 / standard13C - 1 ) * 1000
        return delta13C


    def _get_Delta14C(self, C, C14):
        standard14C = 1e-12 # approximation used in CESM2
        Delta14C = ( C14/C / standard14C - 1 ) * 1000
        return Delta14C


    def _read_data(self, variable):

        if variable not in self._variables:
            raise NotImplementedError

        # Read multi-file dataset
        ds = self.sourcefile_groups[variable].read() # xarray DataSet

        # Select variable at the nearest latitude and longitude
        da = ds[variable].sel(lat=self.lat, lon=self.lon, method='nearest')

        if variable in ('TSOI', 'H2OSOI', 'C14_SOILC_vr', 'SOILC_vr'):
            # Average over the top-bot soil depth interval
            da = self._average_over_depth(da, self.top, self.bot)

        # Make a pandas.Series from the xarray.DataArray
        series = da.to_pandas()

        # The time index is ahead by a month. Fix that.
        dt = np.timedelta64(15,'D') # set the index back by 15 days
        series.index = series.index.to_datetimeindex(unsafe=True) - dt
        series = series.resample('MS').bfill() # resample to month start
        # There is probably a better way of doing this

        series.index.name = 'time'

        return series


    @staticmethod
    def _average_over_depth(da, top, bot):
        """
        Take a weighted average of the data in DataArray `da`
        over its depth dimension on the `top`-`bot` depth interval.
        The depth dimension is either 'levgrnd' or 'levsoi'.
        The data in each depth layer is weighted by the length of the
        overlap between the depth layer and the `top`-`bot` depth range.
        
        Further information on the depth dimension:
        https://bb.cgd.ucar.edu/cesm/threads/cesm2-le-soilliq-and-tsoi-vertical-coordinate-differences.6953/
        https://bb.cgd.ucar.edu/cesm/threads/depths-of-levsoi-as-opposed-to-levgrnd.6212/
        
        Parameters
        ----------
        da : xarray.DataArray
            data to average over its depth dimension
        top : float, int
            top of the depth range over which to integrate (in cm)
        bot : float, int
            bottom of the depth range over which to integrate (in cm)
        
        Returns
        -------
        da : xarray.DataArray
            data averaged over depth
        """

        if da.name not in ('TSOI', 'H2OSOI', 'C14_SOILC_vr', 'SOILC_vr'):
            raise NotImplementedError

        if 'levgrnd' in da.dims:
            depth_dim = 'levgrnd'
        elif 'levsoi' in da.dims:
            depth_dim = 'levsoi'
        else:
            raise ValueError

        # Convert cm to m
        top = top / 100
        bot = bot / 100

        # Soil depth layer numbers
        levsoi = np.arange(20)

        # Midpoint depths (m) of soil layers for TSOI
        levgrnd = np.array([
        1.000000e-02, 4.000000e-02, 9.000000e-02, 1.600000e-01, 2.600000e-01,
        4.000000e-01, 5.800000e-01, 8.000000e-01, 1.060000e+00, 1.360000e+00,
        1.700000e+00, 2.080000e+00, 2.500000e+00, 2.990000e+00, 3.580000e+00,
        4.270000e+00, 5.060000e+00, 5.950000e+00, 6.940000e+00, 8.030000e+00,
        9.795000e+00, 1.332777e+01, 1.948313e+01, 2.887072e+01, 4.199844e+01
        ])

        # Bounds of the soil depth layers
        bounds = reduce(
            lambda bnds, midpoint: bnds + [2*midpoint - bnds[-1]], levgrnd, [0]
        )

        # Index of layers that overlap with our top-bot depth range
        selected_layers_index = [
            i for i, (d1, d2) in enumerate(zip(bounds[:-1], bounds[1:]))
            if d2 > top and d1 < bot
        ]

        # Overlap (in meters) between the selected layers and the top-bot range
        selected_layers_overlap = np.array([
            min(bot, bounds[i+1]) - max(top, bounds[i])
            for i in selected_layers_index
        ])

        # Select data in the depth layers which overlap with the top-bot range
        da = da.isel(**{depth_dim: selected_layers_index})

        # Take weighted average of the variable over the selected depth layers
        weights = xr.DataArray(
            selected_layers_overlap / selected_layers_overlap.sum(),
            coords={depth_dim: da[depth_dim].values}, dims=depth_dim
        )
        da = da.weighted(weights).mean(dim=depth_dim)

        return da
