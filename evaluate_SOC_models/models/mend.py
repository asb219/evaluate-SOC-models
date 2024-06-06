"""
Radiocarbon implementation and python interface for modified source
code (Wang & Brunmayr, 2024) of MEND model (Wang et al., 2022).

* Modified source code of MEND:
    Wang, G., & Brunmayr, A. S. (2024). "Microbial-ENzyme Decomposition
    (MEND) model – GitHub fork asb219/MEND (v1.1 (MEND-new-asb219))".
    Zenodo. https://doi.org/10.5281/zenodo.11065513

* Original source code of MEND:
    Wang, G. (2024). "Microbial-ENzyme Decomposition (MEND) model (commit
    92323c7 in https://github.com/wanggangsheng/MEND)". Zenodo.
    https://doi.org/10.5281/zenodo.10576665

* Associated manuscript:
    Wang, G., et al. (2022). "Soil enzymes as indicators of soil function:
    A step toward greater realism in microbial ecological modeling". Global
    Change Biology, 28(5), 1935–1950. https://doi.org/10.1111/gcb.16036


Copyright (C) 2024  Alexander S. Brunmayr  <asb219@ic.ac.uk>

This file is part of the ``evaluate_SOC_models`` python package, subject to
the GNU General Public License v3 (GPLv3). You should have received a copy
of GPLv3 along with this file. If not, see <https://www.gnu.org/licenses/>.
"""

import subprocess
import numpy as np
import pandas as pd
import scipy.optimize
from numba import njit
import f90nml
from loguru import logger

from data_manager import Data, DataFile, PandasCSVFile
from evaluate_SOC_models.path import SAVEOUTPUTPATH, MENDREPOSITORYPATH
from .base import ModelEvaluationData


__all__ = ['MENDData']


class Fortran90NamelistFile(DataFile):

    def _read(self):
        """Read namelist file and returns `f90nml.Namelist` object."""
        with self.path.open('r') as nml_file:
            return f90nml.read(nml_file)

    def _write(self, namelist, force=True, sort=False):
        """Write namelist to file.
        
        Parameters
        ----------
        namelist : :py:class:`f90nml.Namelist`
            object to be written to file
        force : bool, default True
            overwrite existing file
        sort : bool, default False
            alphabetically sort the namelist
        """
        with self.path.open('w') as nml_file:
            return f90nml.write(namelist, nml_file, force=force, sort=sort)



class MENDOutFile(PandasCSVFile):

    def __init__(self, *args, readonly=True, **kwargs):
        super().__init__(*args, readonly=readonly, **kwargs)

    def _read(self, skiprows=[0,2], sep='\s+', converters='default', **kwargs):
        # Skip row 2 because the first entry is useless/problematic:
        #   in VAR_hour: first entry is initial condition at Hour '0'
        #   in FLX_hour: first entry has problematic values very close to zero
        if converters == 'default':
            converters = {'Hour':
                lambda x: pd.to_datetime(str(int(x)-1), format='%Y%m%d%H')}
        df = super()._read(
            skiprows=skiprows, sep=sep,
            converters=converters, #dtype='object',
            **kwargs
        ).set_index('Hour')
        #df = df.apply(pd.to_numeric, errors='coerce').fillna(0.)
        # coerce too small numbers to NaN and replace them with 0
        return df



class SoilIniDatFile(PandasCSVFile):

    def _read(self):
        with self.path.open('r') as f:
            header = [f.readline().strip() for line in range(2)]
        df = super()._read(sep='\s+', skiprows=[0,1])
        df.columns = [c.strip() for c in df.columns] # '\xa0Value' -> 'Value'
        series = df.set_index('Property')['Value']
        return series, header

    def _write(self, series, header):
        with self.path.open('w') as f:
            for line in header:
                f.write(line + '\n')
        df = series.reset_index('Property')
        df['ID'] = range(1, len(series)+1)
        df = df[['ID', 'Property', 'Value']]
        return super()._write(df, mode='a', sep='\t', index=False)



class MonthlyDatFile(PandasCSVFile):

    def _read(self):
        series = super()._read(
            parse_dates={'time': ['year','month']}
        ).set_index('time').squeeze()
        return series

    def _write(self, series, **kwargs):
        """ series need timestamp index """
        #value_column_name = series.name
        series.name = 'aCaN'
        df = series.to_frame()
        df['year'] = df.index.year
        df['month'] = df.index.month
        df = df[['year', 'month', 'aCaN']]
        return super()._write(df, mode='w', sep='\t', index=False, **kwargs)



class MENDData(ModelEvaluationData):

    model_name = 'MEND'

    datasets = [
        'forcing', 'preindustrial_forcing', 'constant_forcing',
        'spinup_C_forcing', 'spinup_F_forcing', 'realrun_forcing',
        'initial_state', 'initial_F',
        'raw_output', 'output', 'predicted', 'observed', 'error'
    ]

    all_pools = [
        'POM1', 'POM2', 'MOM', 'QOM', 'DOM', 'MBA', 'MBD', 'EP1', 'EP2', 'EM',
        *[f'ENZNm{i}' for i in range(1,7)]
    ]
    fraction_to_pool_dict = {
        'LF': ['POM1','POM2'], # 'DOM', 'MBA', 'MBD'
        'HF': ['MOM','QOM'],
        'bulk': ['POM1', 'POM2', 'MOM', 'QOM', 'DOM', 'MBA', 'MBD']
            # + ['EP1', 'EP2', 'EM', *[f'ENZNm{i}' for i in range(1,7)]]
    }

    _raw_output_stocks = {
        'POM1': 'POMO_C',
        'POM2': 'POMH_C',
        'MOM': 'MOM_C',
        'QOM': 'QOM_C',
        'DOM': 'DOM_C',
        'MBA': 'MBA_C', # active microbes
        'MBD': 'MBD_C', # dormant microbes
        'EP1': 'ENZP1_C', # enzyme for POM1
        'EP2': 'ENZP2_C', # enzyme for POM2
        'EM': 'ENZM_C', # enzyme for MOM
        'ENZNm1': 'ENZNfix_C',
        'ENZNm2': 'ENZNH4_C',
        'ENZNm3': 'ENZNO3_C',
        'ENZNm4': 'ENZNO2_C',
        'ENZNm5': 'ENZNO_C',
        'ENZNm6': 'ENZN2O_C'
    }
    _raw_output_influx = {
        'POM1': 'POMinp1_C',
        'POM2': 'POMinp2_C',
        'DOM': 'DOMinp_C'
    }
    _raw_output_transfer = {
        'POM1': {'DOM': 'POMdec2DOM1_C', 'MOM': 'POMdec2MOM1_C'},
        'POM2': {'DOM': 'POMdec2DOM2_C', 'MOM': 'POMdec2MOM2_C'},
        'MOM':  {'DOM': 'MOM2DOM_C', 'QOM': 'MOM2QOM_C'},
        'QOM':  {'DOM': 'QOM2DOM_C', 'MOM': 'QOM2MOM_C'},
        'DOM':  {'QOM': 'DOM2QOM_C', 'MBA': 'DOM2MBA_C'},
        'MBA':  {'POM1': 'MBA2POM1_C', 'POM2': 'MBA2POM2_C',
                'DOM': 'MBA2DOM_C', 'MBD': 'MBA2MBD_C',
                'EP1': 'MBA2EP1_C', 'EP2': 'MBA2EP2_C', 'EM': 'MBA2EM_C',
                **{f'ENZNm{i}': f'MBA2ENZNm{i}_C' for i in range(1,7)},
                'CO2_growth': 'CO2_growth',
                'CO2_maintn': 'CO2_maintn_MBA',
                'CO2_ovflow': 'CO2_ovflow_MBA'},
        'MBD':  {'MBA': 'MBD2MBA_C',
                'CO2_maintn': 'CO2_maintn_MBD',
                'CO2_ovflow': 'CO2_ovflow_MBD'},
        'EP1':  {'DOM': 'EP2DOM1_C'},
        'EP2':  {'DOM': 'EP2DOM2_C'},
        'EM':   {'DOM': 'EM2DOM_C'},
        **{ f'ENZNm{i}': {'DOM': f'ENZNm2DOM{i}_C'} for i in range(1,7) }
    }


    def __init__(self, entry_name, site_name, pro_name,
            spinup_C=400, spinup_F=1000, auto_remove_iofiles=True,
            *, save_pkl=True, save_csv=False, save_xlsx=False, **kwargs):

        super().__init__(entry_name, site_name, pro_name, save_pkl=save_pkl,
            save_csv=save_csv, save_xlsx=save_xlsx, **kwargs)

        self.spinup_C = int(spinup_C)
        self.spinup_F = int(spinup_F)
        self._auto_remove_iofiles = auto_remove_iofiles

        self.mend_executable_path = next( # choose first matching path
            MENDREPOSITORYPATH.glob('dist/**/mend'))

        self.output_prefix = 'aCaN' # sSite in the template namelist.nml

        run_types = ('spinup_C', 'spinup_F', 'realrun')

        self.input_path = {
            run: self.savedir / ('input_' + run) for run in run_types
        }
        self.output_path = {
            run: self.savedir / ('output_' + run) for run in run_types
        }

        self.namelist_files = {
            run: Fortran90NamelistFile(self.savedir / f'namelist_{run}.nml')
            for run in run_types
        }
        self.namelist_files['template'] = Fortran90NamelistFile(
            MENDREPOSITORYPATH / 'MEND_namelist.nml', readonly=True
        )
        self.output_files = {
            run: {
                outfile: MENDOutFile(
                    path / (self.output_prefix + '_' + outfile + '.out')
                ) for outfile in ('VAR_hour', 'FLX_hour')
            } for run, path in self.output_path.items()
        }
        self.input_files = {
            run: {
                'SOIL_INI_template': SoilIniDatFile(
                    MENDREPOSITORYPATH/'userio'/'inp'/'SOIL_INI.dat',
                    readonly=True
                ),
                'SOIL_INI': SoilIniDatFile(path/'SOIL_INI.dat'),
                'Tsoil': MonthlyDatFile(path/'Tsoil.dat'),
                'Wsoil': MonthlyDatFile(path/'Wsoil.dat'),
                'GPP': MonthlyDatFile(path/'GPP.dat'),
                'NHx': MonthlyDatFile(path/'NHx.dat'),
                'NOy': MonthlyDatFile(path/'NOy.dat')
            } for run, path in self.input_path.items()
        }


    def _process_forcing(self):
        forc = self._forcing['dynamic'][
            ['Tsoil', 'Wsoil', 'GPP_total', 'Delta14Clit', 'NHx', 'NOy']
        ].copy().rename(columns={'GPP_total': 'GPP'})
        forc['Tsoil'] -= 273 # Kelvin -> degrees Celsius
        forc['GPP'] *= 1e3 / 1e4 * (60*60) # gC/m2/s -> mgC/cm2/h
        forc['NHx'] *= 100 * 1e3 / 1e4 / (365/12*24) # gN/m2/month -> mgN/cm2/h
        forc['NOy'] *= 100 * 1e3 / 1e4 / (365/12*24) # gN/m2/month -> mgN/cm2/h
        return forc


    def _process_spinup_C_forcing(self):
        start_year = '1000'
        end_year = str(int(start_year) + self.spinup_C)
        index = pd.date_range(str(start_year), str(end_year), freq='ME',
        unit='s')
        forc_year = self['preindustrial_forcing']
        forc = pd.concat([forc_year] * self.spinup_C, ignore_index=True)
        forc.index = index
        return forc


    def _process_spinup_F_forcing(self):
        index = pd.date_range('1000', '1001', freq='ME', unit='s')
        forc_year = self['preindustrial_forcing']
        forc_year.index = index
        return forc_year


    def _process_realrun_forcing(self):
        return self['forcing']


    def _run(self, run, force=True,
            read_VAR_hour=True, read_FLX_hour=True,
            read_VAR_hour_kwargs={}, read_FLX_hour_kwargs={},
            remove_files=True
        ):

        loginfo = lambda x: self._loginfo(run, x)
        logdebug = lambda x: self._logdebug(run, x)

        output_files = self.output_files[run]

        if force or not output_files['VAR_hour'].exists():
            logdebug('Write namelist file')
            self._write_namelist(run)
            logdebug('Write input files')
            self._write_input(run)
            loginfo('Executing MEND...')
            self._execute_mend(run)
        else:
            logdebug('MEND output already exists, do not re-run')

        if read_VAR_hour:
            logdebug('Read VAR_hour output file')
            VAR_hour = output_files['VAR_hour'].read(**read_VAR_hour_kwargs)
        else:
            VAR_hour = None

        if read_FLX_hour:
            logdebug('Read FLX_hour output file')
            FLX_hour = output_files['FLX_hour'].read(**read_FLX_hour_kwargs)
        else:
            FLX_hour = None

        if remove_files:
            logdebug('Remove MEND input and output files')
            for file_path in self.output_path[run].glob('*'):
                file_path.unlink()
            self.output_path[run].rmdir()
            for file_path in self.input_path[run].glob('*'):
                file_path.unlink()
            self.input_path[run].rmdir()

        return VAR_hour, FLX_hour


    def _process_initial_state(self):
        run = 'spinup_C'
        VAR_hour, FLX_hour = self._run(
            run, force=True, read_VAR_hour=True, read_FLX_hour=False,
            read_VAR_hour_kwargs={'converters': None},
            remove_files=self._auto_remove_iofiles
        )
        initial_state = VAR_hour.iloc[-1]
        self._loginfo(run, 'Done')
        return initial_state


    def _process_initial_F(self):
        run = 'spinup_F'
        VAR_hour, FLX_hour = self._run(
            run, force=True, read_VAR_hour=True, read_FLX_hour=True,
            read_VAR_hour_kwargs={'converters': None},
            read_FLX_hour_kwargs={'converters': None},
            remove_files=self._auto_remove_iofiles
        )

        self._logdebug(run, 'Produce 14C output')

        TC_all = self._get_TC_all(FLX_hour)
        I_all = self._get_I_all(FLX_hour)
        IFin_all = I_all
        ordered_names = [self._raw_output_stocks[p] for p in self.all_pools]
        C0 = VAR_hour[ordered_names].iloc[0].values
        F0 = np.zeros_like(C0) + 0.95
        Ncycles = self.spinup_F
        F = self._integrate_spinup_F(TC_all, I_all, IFin_all, C0, F0, Ncycles)
        initial_F = pd.Series(F, index=self.all_pools)

        self._loginfo(run, 'Done')

        return initial_F


    def _process_raw_output(self):

        initial_F = self['initial_F']

        run = 'realrun'
        VAR_hour, FLX_hour = self._run(
            run, force=True, read_VAR_hour=True, read_FLX_hour=True,
            read_VAR_hour_kwargs={'usecols': {
                'Hour', 'TM_C', 'TOM_C', 'SOM_C', 'POMO_C', 'POMH_C',
                'MOM_C', 'QOM_C', 'DOM_C', 'MBA_C', 'MBD_C',
                'ENZP1_C', 'ENZP2_C', 'ENZM_C', 'ENZNfix_C', 'ENZNH4_C',
                'ENZNO3_C', 'ENZNO2_C', 'ENZNO_C', 'ENZN2O_C',
            }},
            read_FLX_hour_kwargs={'usecols': {
                'Hour', 'TOTout_C',
                'TOTinp_C', 'POMinp1_C', 'POMinp2_C', 'DOMinp_C',
                'POMdec1_C', 'POMdec2_C', 'POMdec2DOM1_C', 'POMdec2DOM2_C',
                'POMdec2MOM1_C', 'POMdec2MOM2_C', 'MOMdec_C', 'MOM2DOM_C',
                'QOM2DOM_C', 'DOM2QOM_C', 'DOM2QOMnet_C', 'QOM2MOM_C',
                'MOM2QOM_C', 'MOM2QOMnet_C', 'DOM2MBA_C',
                'MBA_mortality_C', 'MBA2EP1_C', 'MBA2EP2_C', 'MBA2EM_C',
                'MBA_PM_C', 'EP2DOM1_C', 'EP2DOM2_C', 'EM2DOM_C', 'MBA2DOM_C',
                'MBA2POM1_C', 'MBA2POM2_C', 'MBA2MBD_C', 'MBD2MBA_C',
                'MBA2ENZNm1_C', 'MBA2ENZNm2_C', 'MBA2ENZNm3_C', 'MBA2ENZNm4_C',
                'MBA2ENZNm5_C', 'MBA2ENZNm6_C', 'ENZNm2DOM1_C', 'ENZNm2DOM2_C',
                'ENZNm2DOM3_C', 'ENZNm2DOM4_C', 'ENZNm2DOM5_C', 'ENZNm2DOM6_C',
                'CO2_growth', 'CO2_maintn_MBA', 'CO2_maintn_MBD',
                'CO2_ovflow_MBA', 'CO2_ovflow_MBD'
            }},
            remove_files=self._auto_remove_iofiles
        )
        # self.VAR_hour = VAR_hour
        # self.FLX_hour = FLX_hour

        self._logdebug(run, 'Produce C and 14C output')

        TC_all = self._get_TC_all(FLX_hour)
        I_all = self._get_I_all(FLX_hour)

        ordered_names = [self._raw_output_stocks[p] for p in self.all_pools]
        C0 = VAR_hour[ordered_names].iloc[0].values
        F0 = initial_F.values

        flux_times = FLX_hour.index
        flux_dates = flux_times.strftime('%Y%m')
        flux_dates = pd.to_datetime(flux_dates, format='%Y%m')
        Delta14Cin = self['forcing'].loc[flux_dates, ['Delta14Clit']].values
        Fin = Delta14Cin / 1000 + 1
        IFin_all = I_all * Fin

        C_all = VAR_hour[ordered_names].values
        F_all = self._integrate_realrun_F(TC_all, I_all, IFin_all, C_all, F0)

        cpools = self.all_pools
        c14pools = [p + '_14C' for p in cpools]
        out = pd.DataFrame(
            np.concatenate([C_all, F_all], axis=1),
            columns = cpools + c14pools,
            index = VAR_hour.index
        ).resample('D').mean() # downsample to save on disk space and memory

        self._loginfo(run, 'Done')

        return out


    def _get_Cstocks_Delta14C(self, raw_output, fraction):
        cpools = self.fraction_to_pool_dict.get(fraction, [fraction])
        c14pools = [p + '_14C' for p in cpools]
        F = raw_output[c14pools].values
        C = raw_output[cpools].values
        Cstocks = C.sum(axis=1)
        C14stocks = (F * C).sum(axis=1)
        Delta14C = (C14stocks / Cstocks - 1) * 1000
        depth = self['constant_forcing']['lyr_bot']
        Cstocks *= 1e-3 * depth # mgC/cm3 -> gC/cm2
        return Cstocks, Delta14C


    @staticmethod
    @njit
    def _integrate_spinup_F(TC_all, I_all, IFin_all, C0, F0, Ncycles):
        decay_14C = np.log(2)/5730 / 365 / 24 # per hour
        F = F0.copy()
        for _ in range(Ncycles):
            C = C0.copy()
            FC = F * C
            for TC, I, IFin in zip(TC_all, I_all, IFin_all):
                C += TC.sum(axis=1) + I
                FC += TC.dot(F) + IFin - decay_14C * FC
                F = FC/C
        return F


    @staticmethod
    @njit
    def _integrate_realrun_F(TC_all, I_all, IFin_all, C_all, F0):
        decay_14C = np.log(2)/5730 / 365 / 24 # 1/hour
        F_all = np.empty_like(C_all)
        F = F_all[0] = F0
        for i, (TC, I, IFin, C) in enumerate(
                zip(TC_all, I_all, IFin_all, C_all[:-1])):
            FC = F * C
            FC = FC + TC.dot(F) + IFin - decay_14C * FC
            C = C + TC.sum(axis=1) + I
            F = FC/C
            F_all[i+1] = F
        return F_all


    def _get_TC_all(self, fluxes):
        """Build internal transfer matrix for all time steps."""
        pools = self.all_pools
        Npools = len(pools)
        transfer = self._raw_output_transfer
        zeros = np.zeros(len(fluxes))
        TC_all = np.array([
            [fluxes[transfer[source][sink]].values
            if sink in transfer[source] else zeros
            for source in pools
            ] for sink in pools
        ])
        #TC_all = np.where(TC_all>=0, TC_all, 0.)
        TC_all[range(Npools), range(Npools)] -= [
            fluxes[list(transfer[source].values())].sum(axis=1)
            for source in pools
        ]
        TC_all = TC_all.transpose((2,0,1))
        return TC_all


    def _get_I_all(self, fluxes):
        """Build influx vector for all time steps."""
        pools = self.all_pools
        influx = self._raw_output_influx
        zeros = np.zeros(len(fluxes))
        I_all = np.array([
            fluxes[influx[sink]].values if sink in influx else zeros
            for sink in pools
        ]).T
        return I_all


    def _write_input(self, run):
        """Write forcing data files and initial condition file to run MEND."""

        assert run in {'realrun', 'spinup_C', 'spinup_F'}

        cforc = self['constant_forcing']
        dforc = self[run + '_forcing']

        input_files = self.input_files[run]

        input_files['Tsoil'].write(dforc['Tsoil'], float_format='%.1f')
        input_files['Wsoil'].write(dforc['Wsoil'], float_format='%.2f')
        input_files['GPP'].write(dforc['GPP'], float_format='%.2e')
        input_files['NHx'].write(dforc['NHx'], float_format='%.2e')
        input_files['NOy'].write(dforc['NOy'], float_format='%.2e')

        soil_ini, header = input_files['SOIL_INI_template'].read()
        soil_ini['Depth'] = cforc['lyr_bot'] # cm
        soil_ini['Sand'] = cforc['sand'] / 100 # percent -> fraction
        soil_ini['Clay'] = cforc['clay'] / 100 # percent -> fraction

        if run == 'spinup_C':
            soil_ini['SOC'] += 1 # mgC/cm3
            soil_ini['POC'] += 1 # mgC/cm3
            input_files['SOIL_INI'].write(soil_ini, header)
            return

        initial_state = self['initial_state']

        soil_ini['CN_EP1'] = initial_state['ENZP1_CN']
        soil_ini['CN_EP2'] = initial_state['ENZP2_CN']
        soil_ini['CN_EM'] = initial_state['ENZM_CN']
        soil_ini['fQOM'] = initial_state['QOM_C'] / initial_state['MOM_C']
        soil_ini['SOC'] = initial_state['SOM_C']
        soil_ini['POC'] = initial_state['POMT_C']
        soil_ini['MOC'] = initial_state['MOMT_C'] # = 'MOM_C' + 'QOM_C'
        soil_ini['DOC'] = initial_state['DOM_C']
        soil_ini['MBC'] = initial_state['MB_C']
        soil_ini['EP1'] = initial_state['ENZP1_C']
        soil_ini['EP2'] = initial_state['ENZP2_C']
        soil_ini['EM'] = initial_state['ENZM_C']
        soil_ini['CN_SOM'] = initial_state['SOM_CN']
        soil_ini['CN_POM'] = initial_state['POMT_CN']
        soil_ini['CN_MOM'] = initial_state['MOMT_CN']
        soil_ini['CN_DOM'] = initial_state['DOM_CN']
        soil_ini['CN_MB'] = initial_state['MB_CN']
        soil_ini['NH4'] = initial_state['NH4']
        soil_ini['NO3'] = initial_state['NO3']

        input_files['SOIL_INI'].write(soil_ini, header)

        return


    def _write_namelist(self, run):
        """Write parameter values into namelist file to run MEND."""

        assert run in {'realrun', 'spinup_C', 'spinup_F'}

        cforc = self['constant_forcing']
        dforc = self[run + '_forcing']
        input_files = self.input_files[run]

        nml = self.namelist_files['template'].read()
        config = nml['mend_config']

        config['ssMEND'] = 'CN'

        # Site name (used as prefix for output file names)
        config['sSite'] = self.output_prefix

        # Kinetics
        config['siKinetics'] = 0 # official model uses 0
        # 0:Michaelis-Menten & MOM decomposition
        # 10: M-M & QOM decomposition

        # Biome type
        land_cover = cforc['pro_land_cover']
        if land_cover in ('cultivated', 'rangeland/grassland'):
            config['sBIOME'] = 'MGC' # Mesic Grassland & Cropland
        elif land_cover == 'shrubland':
            config['sBIOME'] = 'ASM' # Arid/Semiarid/Mediterranean
        elif land_cover == 'forest':
            if cforc['pro_lc_leaf_type'] == 'needleleaf':
                config['sBIOME'] = 'MCF' # Mesic Conifer Forest
            else:
                config['sBIOME'] = 'MDF' # Mesic Deciduous Forest
        else:
            raise NotImplementedError

        # General stuff
        config['sSOM'] = 'SOL' # intact soil
        if not np.isnan(cforc['site_elevation']):
            config['Altitude'] = cforc['site_elevation'] # meters
        config['Dir_Input'] = str(self.input_path[run]) + '/'
        config['Dir_Output'] = str(self.output_path[run]) + '/'
        config['ssDate_beg_all'] = dforc.index[0].strftime('%Y%m01')
        config['ssDate_end_all'] = dforc.index[-1].strftime('%Y%m31')
        config['ssDate_beg_sim'] = config['ssDate_beg_all']
        config['ssDate_end_sim'] = config['ssDate_end_all']

        # Initial condition
        config['sSOIL_INI_file'] = input_files['SOIL_INI'].filename

        # Soil temperature forcing data
        config['ifdata_ST'] = 1
        config['sUnits_ST'] = 'C' # must be 'C'
        config['step_ST'] = 'monthly'
        config['nfile_ST'] = 1
        config['sfilename_ST'] = input_files['Tsoil'].filename

        # Soil moisture forcing data
        config['ifdata_SM'] = 1
        config['sUnits_SM'] = 'none' # can be 'percent' or 'none' (m3/m3)
        config['step_SM'] = 'monthly'
        config['nfile_SM'] = 1
        config['sfilename_SM'] = input_files['Wsoil'].filename
        max_Wsoil_1850_2014 = self['realrun_forcing']['Wsoil'].max() * 1.01
        config['vg_SWCsat'] = max(config['vg_SWCsat'], max_Wsoil_1850_2014)

        # Carbon input
        avg_GPP_1850_2014 = self['realrun_forcing']['GPP'].mean()
        avg_GPP_1850_2014 *= 24 # mgC/cm2/h -> mgC/cm2/d (must be these units)
        config['GPPref'] = avg_GPP_1850_2014 # "Average GPP or litter input"
        config['ifdata_type1'] = 1
        config['sUnits_type1'] = 'mgC-cm2-h' # must be 'mgC-cm2-h'
        config['step_type1'] = 'monthly'
        config['nfile_type1'] = 1
        config['sfilename_type1'] = input_files['GPP'].filename

        # Soil pH
        config['ifdata_pH'] = 0
        config['sUnits_pH'] = 'pH'
        config['spH_constant'] = cforc['ph_h2o']

        # NH4 input
        config['ifdata_NH4'] = 1
        config['sUnits_NH4'] = 'mgN-cm2-h' # must be 'mgN-cm2-h'
        config['step_NH4'] = 'monthly'
        config['nfile_NH4'] = 1
        config['sfilename_NH4'] = input_files['NHx'].filename

        # NO3 input
        config['ifdata_NO3'] = 1
        config['sUnits_NO3'] = 'mgN-cm2-h' # must be 'mgN-cm2-h'
        config['step_NO3'] = 'monthly'
        config['nfile_NO3'] = 1
        config['sfilename_NO3'] = input_files['NOy'].filename

        # Do not calibrate anything
        config['Cali_Calibrate'] = [0] * len(config['Cali_Calibrate'])

        # Use best parameter values
        config['Pinitial'] = [ # from original MEND_namelist.nml
            0.90000000, 0.01000000, 0.20000000, 0.30000000 , 50.00000000,
            60.00000000, 10.00000000, 999.00000000, 1.50000000, 6.00000000,
            0.00600000, 0.00012289, 0.00147568, 4.50361918, 0.75000000,
            0.50000000, 0.00425194, 0.05012233, 0.00010034, 0.20109320,
            0.00520677, 1.80000000, 0.01030726, 0.00100000, 0.46000000,
            0.39000000, 3.38000000, 0.10000000, 0.00018000, 0.00041000,
            0.06350410, 185.28188371, 0.86952628, 1.00000000, 0.10000000,
            0.00120000, 0.00180000, 0.00180000, 0.00003296, 0.00120000,
            0.00180000, 0.02000000, 0.50000000, 0.00574420,   100.00000000,
            0.00000500, 0.00100000
        ]

        # Write namelist to file
        self.namelist_files[run].write(nml)

        return


    def _execute_mend(self, run):
        """
        Execute "mend". This will write files into the output directory.

        Returns
        -------
        int
            exit code of "mend", zero means success

        Raises
        ------
        subprocess.CalledProcessError
            raised by subprocess.check_call if exit code is not zero
        """
        nml_file_path = self.namelist_files[run].path
        command = f'"{self.mend_executable_path}" "{nml_file_path}"'
        if run == 'spinup_C':
            command += ' spinup'
        return subprocess.check_call(command, shell=True)


    def _loginfo(self, run, message):
        logger.info(f'{self} {run.upper()} : {message}')

    def _logdebug(self, run, message):
        logger.debug(f'{self} {run.upper()} : {message}')
