import pandas as pd

from data_manager import Data

from evaluate_SOC_models.observed_data import ObservedData
from evaluate_SOC_models.forcing_data import ForcingData
from evaluate_SOC_models.path import SAVEPATH


__all__ = ['ModelEvaluationData']


class ModelEvaluationData(Data):

    model_name = 'ModelTemplate'

    datasets = [
        'forcing', 'preindustrial_forcing', 'constant_forcing',
        'raw_output', 'output', 'predicted', 'observed', 'error'
    ]
    all_pools = []
    all_fractions = ['HF', 'LF']
    predicted_columns = [
        'soc', 'bulk_14c', 'HF_14c', 'LF_14c', 'HF_c_perc', 'LF_c_perc'
    ]

    def __init__(self, entry_name, site_name, pro_name, **kwargs):

        assert all(d in self.datasets for d in ModelEvaluationData.datasets)

        model_name = self.model_name

        savedir = SAVEPATH / entry_name / site_name / pro_name / model_name
        name = model_name.lower()
        description = 'Model evaluation data for ' + model_name
        super().__init__(savedir, name, description, **kwargs)

        self.entry_name = entry_name
        self.site_name = site_name
        self.pro_name = pro_name

        self._forcing = ForcingData(entry_name, site_name, pro_name)
        self._observed = ObservedData(entry_name, site_name, pro_name)


    def _process_forcing(self):
        raise NotImplementedError


    def _process_preindustrial_forcing(self):
        preindustrial_forcing = self['forcing'].loc['1850':'1879']
        avg_preindustrial_forcing = preindustrial_forcing.groupby(
            lambda idx: idx.month*10000 + idx.day*100 + idx.hour
        ).mean()
        return avg_preindustrial_forcing


    def _process_constant_forcing(self):
        return self._forcing['constant']


    def _process_raw_output(self):
        raise NotImplementedError


    def _get_Cstocks_Delta14C(self, output, fraction):
        raise NotImplementedError


    def _process_output(self):

        raw_output = self['raw_output']
        output = pd.DataFrame(index=raw_output.index, dtype='float')

        Cstocks_bulk, Delta14C = self._get_Cstocks_Delta14C(raw_output, 'bulk')
        output['soc'] = Cstocks_bulk
        output['bulk_14c'] = Delta14C

        for frac in self.all_fractions + self.all_pools:
            Cstocks, Delta14C = self._get_Cstocks_Delta14C(raw_output, frac)
            output[frac+'_c_perc'] = Cstocks / Cstocks_bulk * 100
            output[frac+'_14c'] = Delta14C

        return output


    def _process_predicted(self):
        output = self['output'][self.predicted_columns].copy()
        dates = self._observed.data.index
        predicted = output.reindex(index=dates, method='nearest')
        return predicted


    def _process_observed(self):
        observed = self._observed.data[self.predicted_columns]
        return observed


    def _process_error(self):
        return self['predicted'] - self['observed']
