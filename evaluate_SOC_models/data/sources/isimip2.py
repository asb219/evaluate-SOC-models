"""
Nitrogen deposition data from the ISIMIP repository https://data.isimip.org

Citation for nitrogen deposition data:
    Tian, H., et al. (2018): "The Global N2O Model Intercomparison Project".
    Bulletin of the American Meteorological Society, 99, 1231–1251.
    https://doi.org/10.1175/BAMS-D-17-0212.1

Citation for ISIMIP repository:
    Rosenzweig, C., et al. (2017). "Assessing inter-sectoral climate change
    risks: the role of ISIMIP". Environmental Research Letters, 12, 010 301.
    https://doi.org/10.1088/1748-9326/12/1/010301
    https://data.isimip.org


Copyright (C) 2024  Alexander S. Brunmayr  <asb219@ic.ac.uk>

This file is part of the ``evaluate_SOC_models`` python package, subject to
the GNU General Public License v3 (GPLv3). You should have received a copy
of GPLv3 along with this file. If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import pandas as pd

from data_manager import Data, XarrayNetCDFFile, FileFromURL

from evaluate_SOC_models.path import DOWNLOADPATH, DATAPATH


__all__ = [
    'ISIMIP2aNitrogenDepositionInputFile',
    'ISIMIP2aNitrogenDepositionData'
]


class ISIMIP2aNitrogenDepositionInputFile(XarrayNetCDFFile, FileFromURL):
    """
    Global NHx or NOy deposition 1860-2016 model input data file

    Inter-Sectoral Impact Model Intercomparison Project (ISIMIP)

    https://www.isimip.org/gettingstarted/input-data-bias-adjustment/details/24/
    Input data set: Nitrogen deposition

    ISIMIP2a: Gridded NHx and NOy deposition simulated by NCAR
    Chemistry-Climate Model Initiative (CCMI) during 1860-2014
    at monthly time step. Nitrogen deposition data was interpolated
    to 0.5° by 0.5° by the nearest grid. Data in 2015 and 2016
    are taken from 2014 as it is assumed to be same.

    Units: gN/m2/month

    Data source: Hanqin Tian and Jia Yang, 2018,
    https://doi.org/10.1175/BAMS-D-17-0212.1
    """

    def __init__(self, species):

        if species=='NHx':
            filename = "ndep_nhx_histsoc_monthly_1860_2016.nc4"
        elif species=='NOy':
            filename = "ndep_noy_histsoc_monthly_1860_2016.nc4"
        else:
            raise ValueError()

        savedir = DOWNLOADPATH/'ISIMIP'

        DL_link = (
            "https://files.isimip.org/ISIMIP2a/InputData/"
            "landuse_humaninfluences/n-deposition/histsoc/"
            + filename
        )

        super().__init__(savedir / filename, DL_link)


    def _read(self, **kwargs):
        # Can't decode time units "months since 1860-1-1 00:00:00"
        # Solution: https://stackoverflow.com/a/55686749
        kwargs['decode_times'] = False
        ds = super()._read(**kwargs)
        months, since, ref_date, ref_time = ds.time.units.split()
        ds['time'] = pd.date_range(
            start=ref_date, periods=ds.sizes['time'], freq='MS')
        ds['time'] = ds['time'].assign_attrs(
            standard_name='time', long_name='time', axis='T')
        return ds



class ISIMIP2aNitrogenDepositionData(Data):
    " Inter-Sectoral Impact Model Intercomparison Project (ISIMIP) "
    " https://www.isimip.org/ "

    datasets = ['NHx', 'NOy']

    def __init__(self, lat, lon, *,
            save_pkl=True, save_csv=False, save_xlsx=False):

        lat = np.round(float(lat),2)
        lon = np.round(float(lon),2)

        identifier = f'lat{lat:+06.2f}_lon{lon:+07.2f}'
        savedir = DATAPATH / 'ISIMIP' / identifier
        name = 'ISIMIP2a_Ndep'
        description = (
            'ISIMIP2a nitrogen deposition (gN/m2/month) data for '
            f'lat,lon = {lat:.2f},{lon:.2f}.'
        )

        super().__init__(savedir, name, description,
            save_pkl=save_pkl, save_csv=save_csv, save_xlsx=save_xlsx)

        self.lat = lat
        self.lon = lon
        self.sourcefiles = {
            'nhx': ISIMIP2aNitrogenDepositionInputFile('NHx'),
            'noy': ISIMIP2aNitrogenDepositionInputFile('NOy')
        }

    def _process_NHx(self):
        return self._read_data('nhx') # gN/m2/month

    def _process_NOy(self):
        return self._read_data('noy') # gN/m2/month

    def _read_data(self, variable):
        ds = self.sourcefiles[variable].read()
        da = ds[variable].sel(lat=self.lat, lon=self.lon, method='nearest')
        series = da.to_pandas()
        return series
