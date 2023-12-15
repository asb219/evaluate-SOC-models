import pandas as pd
import scipy.optimize
from loguru import logger

from evaluate_SOC_models.data_manager import Data

from evaluate_SOC_models.observed_data import ObservedData
from evaluate_SOC_models.forcing_data import ForcingData
from evaluate_SOC_models.path import SAVEOUTPUTPATH


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

    def __init__(self, entry_name, site_name, pro_name, *,
            savedir=None, name=None, description=None, **kwargs):

        assert all(d in self.datasets for d in ModelEvaluationData.datasets)

        self.entry_name = entry_name
        self.site_name = site_name
        self.pro_name = pro_name

        self._forcing = ForcingData(entry_name, site_name, pro_name)
        self._observed = ObservedData(entry_name, site_name, pro_name)

        model_name = self.model_name

        if savedir is None:
            savedir = SAVEOUTPUTPATH / entry_name / site_name / pro_name / model_name

        if name is None:
            name = model_name.lower()

        if description is None:
            description = 'Model evaluation data for ' + model_name

        super().__init__(savedir, name, description, **kwargs)


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


    def _find_steady_state(self, func, x0, bounds=(None,None), name='C'):
        """
        Find a steady state solution `x` such that `func(x) = 0` with
        first guess `x0`. First try with `scipy.optimize.fsolve(func, x0)`,
        but if the result is out of bounds, then try again with
        `scipy.optimize.least_squares(func, x0, bounds=bounds)`.
        
        Parameters
        ----------
        func
            function taking a vector as its sole argument and returning
            a vector of same length
        x0 : np.ndarray
            first guess for the steady state solution
        bounds : (lo, hi), default (None, None)
            lower and upper bounds the solution vector; `lo` and `hi` can
            be `None`, scalar, or np.ndarray vector with length of `x0`
        name : str, default 'C'
            name of the quantity for which to find a steady state
        
        Returns
        -------
        x : np.ndarray or None
            steady state solution vector
        success : bool
            whether `x` is a good steady state solution
        """

        x = scipy.optimize.fsolve(func, x0)
        lo, hi = bounds
        success = (lo is None or all(x > lo)) and (hi is None or all(x < hi))

        if not success:
            logger.debug(
                f'scipy.optimize.fsolve failed to find steady state'
                f' of {name} for {self}.'
                ' Falling back to scipy.optimize.least_squares.'
            )

            sol = scipy.optimize.least_squares(func, x0, bounds=bounds)
            x = sol.x
            success = sol.success

            if not success:
                logger.warning(
                    f'scipy.optimize.least_squares failed to find {name}'
                    f' steady state for {self}.'
                    f' STATUS: {sol.status}. MESSAGE: {sol.message}'
                )

        return x, success


    def __repr__(self):
        info = f'("{self.entry_name}", "{self.site_name}", "{self.pro_name}")'
        return self.__class__.__name__ + info

    def __str__(self):
        return self.__repr__()
