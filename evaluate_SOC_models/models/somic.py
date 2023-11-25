from pathlib import Path
import subprocess

import pandas as pd
import numpy as np

from data_manager import PandasCSVFile
from evaluate_SOC_models.model_data import ModelEvaluationData



class SOMicData(ModelEvaluationData):

    model_name = 'SOMic'

    datasets = [
        'forcing', 'preindustrial_forcing', 'constant_forcing',
        'all_data_spinup', 'exp_const_spinup', 'raw_output_spinup',
        'all_data', 'exp_const', 'raw_output',
        'output', 'predicted', 'observed', 'error'
    ]

    all_pools = [
        'spm', # soluble plant matter
        'ipm', # insoluble plant matter
        'doc', # dissolved organic carbon
        'mb', # microbial biomass
        'mac', # mineral associated carbon
        #'soc' # total soil organic carbon (sum of all pools)
    ]
    fraction_to_pool_dict = {
        'LF': ['spm', 'ipm'], # 'mb', 'doc'
        'HF': ['mac'], # 'mb'
        'bulk': ['soc']
    }

    def __init__(self, entry_name, site_name, pro_name, spinup=5000, # years
            auto_remove_csv=True, use_fraction_modern=True,
            *, save_pkl=False, save_csv=False, save_xlsx=False):

        super().__init__(entry_name, site_name, pro_name,
            save_pkl=save_pkl, save_csv=save_csv, save_xlsx=save_xlsx)

        self.spinup = spinup
        self._auto_remove_csv = auto_remove_csv
        self.use_fraction_modern = use_fraction_modern # instead of 14C-age

        self.R_interface_files = {
            'all_data_spinup': PandasCSVFile(self.savedir/'all_data_spinup.csv'),
            'exp_const_spinup': PandasCSVFile(self.savedir/'exp_const_spinup.csv'),
            'output_spinup': PandasCSVFile(self.savedir/'output_spinup.csv', readonly=True),
            'all_data': PandasCSVFile(self.savedir/'all_data.csv'),
            'exp_const': PandasCSVFile(self.savedir/'exp_const.csv'),
            'output': PandasCSVFile(self.savedir/'output.csv', readonly=True)
        }


    def _process_forcing(self):
        forc = self._forcing['dynamic'][
            ['Tsoil', 'Wsoil', 'NPP', 'Delta14Clit']
        ].copy() # data is already monthly
        forc['Tsoil'] -= 273.15 # convert Kelvin to degrees Celsius
        return forc


    def _process_raw_output_spinup(self):
        return self._produce_output(is_spinup=True)


    def _process_raw_output(self):
        return self._produce_output(is_spinup=False)


    def _process_exp_const_spinup(self):

        init_14c = 1 if self.use_fraction_modern else 0

        exp_const = pd.Series({
            'use_fraction_modern_instead_of_14c_age':
                int(self.use_fraction_modern),
            'init_spm_14c': init_14c,
            'init_ipm_14c': init_14c,
            'init_doc_14c': init_14c,
            'init_mb_14c': init_14c,
            'init_mac_14c': init_14c,
            'clay': self['constant_forcing']['clay']
        }, dtype='float').to_frame().transpose()

        return exp_const


    def _process_exp_const(self):

        initial_condition = self['raw_output_spinup'].iloc[-1]

        exp_const = pd.Series({
            'use_fraction_modern_instead_of_14c_age':
                int(self.use_fraction_modern),
            'init_spm_14c': initial_condition['spm.14c'],
            'init_ipm_14c': initial_condition['ipm.14c'],
            'init_doc_14c': initial_condition['doc.14c'],
            'init_mb_14c': initial_condition['mb.14c'],
            'init_mac_14c': initial_condition['mac.14c'],
            'clay': self['constant_forcing']['clay']
        }, dtype='float').to_frame().transpose()

        return exp_const


    def _process_all_data_spinup(self):

        # Spin up over pre-industrial forcing data
        all_data_preindustrial = self._build_all_data_template(is_spinup=True)
        all_data = pd.concat(
            [all_data_preindustrial] * self.spinup, axis=0, ignore_index=True
        ) # repeat typical pre-industrial year for the number of spinup years

        idx0 = all_data.index[0]

        # Set initial condition of 1 MgC/ha for carbon stocks of all pools
        all_data.loc[idx0, ['spm', 'ipm', 'doc', 'mb', 'mac']] = 1

        # Set initial condition of -25 permille for d13C of all pools
        all_data.loc[idx0, 'soc.d13c'] = -25

        return all_data


    def _process_all_data(self):

        all_data = self._build_all_data_template(is_spinup=False)

        # Set initial condition for carbon stocks and d13C of carbon pools
        idx0 = all_data.index[0]
        columns = ['spm', 'ipm', 'doc', 'mb', 'mac', 'soc', 'soc.d13c']
        all_data.loc[idx0, columns] = self['raw_output_spinup'][columns].iloc[-1]

        return all_data


    def _build_all_data_template(self, is_spinup):

        # Required columns necessary to run SOMic
        required_columns = [
            'time', # integers, used to get simulation length and for output
            'spm', # MgC/ha, soluble plant matter, only first row matters
            'ipm', # MgC/ha, insoluble plant matter, only first row matters
            'doc', # MgC/ha, dissolved organic carbon, only first row matters
            'mb',  # MgC/ha, microbial biomass, only first row matters
            'mac', # MgC/ha, mineral-associated carbon, only first row matters
            'soc', # MgC/ha, total soil organic carbon, only first row matters
            'added.spm', # MgC/ha/month, carbon input into spm pool
            'added.ipm', # MgC/ha/month, carbon input into ipm pool
            'added.doc', # MgC/ha/month, carbon input into doc pool
            'added.mb',  # MgC/ha/month, carbon input into mb pool
            'added.mac', # MgC/ha/month, carbon input into mac pool
            'temp', # Celsius, soil temperature
            'h2o', # m/m (= mm3/mm3 hopefully), soil moisture
            'a', # temperature rate modifier
            'c', # soil cover rate modifier
            'added.d13c', # permille, d13C of carbon input
            'soc.d13c', # permille, initial d13C of all pools
            'add_14c', # in months, 14C-age of carbon input
            'velocity' # DOC leaching velocity
        ]

        # The below columns are used to calculate rate modifiers for moisture
        # and microbial biomass when use_atsmd = 1. However, we set use_atsmd
        # = 0 in somic.r, so SOMic uses "h2o" to calculate the rate modifiers.
        required_but_unused_columns = [
            'atsmd', 'cover', 'precip', 'pet'
        ]

        # Useless, but they're in the default all.data of the SOMic R package
        useless_columns = [
            'measured.d14c', 'clay', 'depth', 'lat', 'long', 'max_tsmd',
            'measured.soc', 'fym', 'litter', 'sat', 'mic', 'spm.d13c',
            'ipm.d13c', 'doc.d13c', 'mb.d13c', 'mac.d13c', 'co2.d13c'
        ]

        # Load forcing data
        dforc = self['preindustrial_forcing' if is_spinup else 'forcing']
        cforc = self['constant_forcing']

        all_data = dforc.rename(columns={'Tsoil': 'temp', 'Wsoil': 'h2o'})

        # Fill carbon pools with 0 MgC/ha
        all_data[['spm', 'ipm', 'doc', 'mb', 'mac', 'soc']] = 0

        # Partition inputs according to section 1.2 of supplement
        land_cover = cforc['pro_land_cover']
        all_data['cover'] = 0 if land_cover=='bare' else 1
        if land_cover in ('forest','shrubland'):
            spm_ipm_ratio = 0.25
        elif land_cover == 'rangeland/grassland':
            spm_ipm_ratio = 0.67
        elif land_cover == 'cultivated':
            spm_ipm_ratio = 1.44
        else:
            # pro_land_cover is 'bare' or other
            raise ValueError(f"Can't handle land cover '{land_cover}'.")

        # Add input data, converting gC/m2/s to MgC/ha/month
        NPP = dforc['NPP'] * 1e-6 * 100**2 * (30*24*60*60)
        all_data['added.ipm'] = NPP / (1 + spm_ipm_ratio)
        all_data['added.spm'] = NPP * spm_ipm_ratio / (1 + spm_ipm_ratio)
        all_data['added.mb'] = 0
        all_data['added.mac'] = 0
        all_data['added.doc'] = 0

        # Rate modifying factor for temperature from eq 3 in supplement
        ft = 4.99 ; Tmax = 45. ; Topt = 35.
        r = (Tmax - all_data['temp']) / (Tmax - Topt)
        all_data['a'] = ft * r**0.2 * np.exp(0.2/2.63 * (1 - r**2.63))

        # Rate modifying factor for soil cover, from eq 6 in supplement
        all_data['c'] = np.where(all_data['cover'], 0.6, 1.)

        # Set constant d13C for carbon input, and set initial state of SOC d13C
        all_data['added.d13c'] = -25
        all_data['soc.d13c'] = -25

        # Set 14C age of carbon input
        input_F14C = dforc['Delta14Clit'] / 1000 + 1
        if self.use_fraction_modern:
            all_data['add_14c'] = input_F14C
        else:
            input_age = - np.log(input_F14C) * 8267 * 12 # in months
            all_data['add_14c'] = input_age

        # Set DOC leaching velocity to zero
        all_data['velocity'] = 0 # that's what Woolf & Lehmann (2019) did too

        # Transform time into an integer and add it as a column
        if not is_spinup:
            all_data.index = all_data.index.strftime('%Y%m').astype('int')
        all_data.index.name = 'time'
        all_data = all_data.reset_index('time')

        # Set values of required but unused columns ('cover' is already set)
        all_data[['atsmd', 'precip', 'pet']] = 0

        columns = required_columns + required_but_unused_columns

        return all_data[columns].copy()


    def _produce_output(self, is_spinup):

        postfix = '_spinup' if is_spinup else ''

        all_data_file = self.R_interface_files['all_data' + postfix]
        exp_const_file = self.R_interface_files['exp_const' + postfix]
        output_file = self.R_interface_files['output' + postfix]

        all_data_file.write(self['all_data' + postfix], index=False)#, float_format='%.4e')
        exp_const_file.write(self['exp_const' + postfix], index=False)#, float_format='%.1f')

        self._run_R(all_data_file.path, exp_const_file.path, output_file.path)

        output = output_file.read(index_col='time', date_format='%Y%m')

        if self._auto_remove_csv:
            all_data_file.remove(ask=False)
            exp_const_file.remove(ask=False)
            output_file.remove(ask=False)

        return output


    def _run_R(self, all_data_path, exp_const_path, output_path):
        somic_program_path = Path(__file__).parent.resolve() / 'somic.r'
        cmdline_args = f'"{all_data_path}" "{exp_const_path}" "{output_path}"'
        command = f'Rscript "{somic_program_path}" ' + cmdline_args
        return subprocess.call(command, shell=True)


    def _get_Cstocks_Delta14C(self, output, fraction):
        cpools = self.fraction_to_pool_dict.get(fraction, [fraction])
        c14pools = [p + '.14c' for p in cpools]
        Cstocks = output[cpools].sum(axis=1)
        if not self.use_fraction_modern:
            ages = output[c14pools].values # ages in months
            decay14C = 1./(8267*12) # radioactive decay rate of 14C per month
            FMs = np.exp(-decay14C * ages) # fractions modern
        else:
            FMs = output[c14pools].values # fractions modern
        FM = (FMs * output[cpools].values).sum(axis=1) / Cstocks
        Delta14C = (FM - 1) * 1000
        Cstocks *= 1e-2 # MgC/ha -> gC/cm2
        return Cstocks, Delta14C
