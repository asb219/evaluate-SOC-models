"""
Methods for running all models on all profiles and processing model output.

Copyright (C) 2024  Alexander S. Brunmayr  <asb219@ic.ac.uk>

This file is part of the ``evaluate_SOC_models`` python package, subject to
the GNU General Public License v3 (GPLv3). You should have received a copy
of GPLv3 along with this file. If not, see <https://www.gnu.org/licenses/>.
"""

import multiprocessing
import numpy as np
import pandas as pd
from loguru import logger

from evaluate_SOC_models.data.observed import SelectedISRaDData
from evaluate_SOC_models.data.forcing import ForcingData
from evaluate_SOC_models.models import \
    MENDData, MillennialData, SOMicData, CORPSEData, MIMICSData


__all__ = [
    'run_model',
    'run_all_models_all_profiles',
    'get_results',
    'get_results_all_profiles',
    'get_all_results',
    'get_bias_and_rmse'
]


SORTED_MODEL_NAMES = ['MEND', 'Millennial', 'SOMic', 'CORPSE', 'MIMICS']
SORTED_VARIABLE_NAMES = \
    ['soc', 'bulk_14c', 'LF_14c', 'HF_14c', 'LF_c_perc', 'HF_c_perc']


def get_all_models():
    return (MENDData, MillennialData, SOMicData, CORPSEData, MIMICSData)

def get_all_profiles():
    return SelectedISRaDData().data.index


MEND_fails = {
    ('Heckman_2018', 'HI_Andisol', 'WPL1204'),
    ('Lemke_2006', 'Solling', 'Solling_DO_F1'),
    ('Lemke_2006', 'Solling', 'Solling_DO_F2'),
    ('Lemke_2006', 'Solling', 'Solling_DO_F3'),
    #('McFarlane_2013', 'MA Harvard Forest', 'H4'), # works now, but 14C fails
    #('McFarlane_2013', 'MI-Coarse UMBS', '60M'), # works on my pc
    #('McFarlane_2013', 'MI-Coarse UMBS', 'C7'), # works now, but 14C fails
    #('McFarlane_2013', 'MI-Coarse UMBS', 'D4'), # works now, but 14C fails
    #('McFarlane_2013', 'MI-Coarse UMBS', 'G3'), # works on my pc
    ('McFarlane_2013', 'MI-Coarse UMBS', 'O2'),
    ('Schrumpf_2013', 'Hesse', 'Hesse.1'),
    #('Schrumpf_2013', 'Laqueuille', 'Laqueuille.1') # works now
}
MEND_runs_but_output_is_problematic = {
    ('Rasmussen_2018', 'GRpp', 'GRpp'), # negative SOC stocks
    ('Rasmussen_2018', 'GRwf', 'GRwf'), # negative SOC stocks
    ('Schrumpf_2013', 'Norunda', 'Norunda.1') # bad C repartition
}
MEND_C_works_but_14C_fails = {
    ('McFarlane_2013', 'MA Harvard Forest', 'H4'), # promoted from MEND_fails
    ('McFarlane_2013', 'MI-Coarse UMBS', 'C7'), # promoted from MEND_fails
    ('McFarlane_2013', 'MI-Coarse UMBS', 'D4'), # promoted from MEND_fails
    ('McFarlane_2013', 'Mi-Fine Colonial Point', 'C1'),
    ('McFarlane_2013', 'Mi-Fine Colonial Point', 'C4'),
    ('McFarlane_2013', 'Mi-Fine Colonial Point', 'C5')
}
MEND_bad_14C_initial_condition_but_still_runs_nicely_for_the_remaining_time = {
    ('McFarlane_2013', 'MA Harvard Forest', 'H5'),
    ('McFarlane_2013', 'NH Bartlett Forest', 'B1'),
    ('McFarlane_2013', 'NH Bartlett Forest', 'B2'),
    ('McFarlane_2013', 'NH Bartlett Forest', 'B3'),
    ('McFarlane_2013', 'NH Bartlett Forest', 'B4'),
    #('McFarlane_2013', 'NH Bartlett Forest', 'B5'), # good now
    ('Schrumpf_2013', 'Laqueuille', 'Laqueuille.1') # promoted from MEND_fails
}
MEND_excluded_profiles = MEND_fails | MEND_runs_but_output_is_problematic


def run_model(model, profile, *, save_pkl=True, force_rerun=False):
    """
    Parameters
    ----------
    model : evaluate_SOC_models.ModelEvaluationData subclass
        class of model to run on profile
    profile : tuple(str, str, str)
        3-tuple of ISRaD profile index (entry_name, site_name, pro_name)
    save_pkl : bool, default True
        pickle model input/output so no need to re-run next time
    force_rerun : bool, default False
        if True, delete all model input/output files and re-run model;
        if False, do not re-run if pickled model output already exists
    
    Returns
    -------
    m : evaluate_SOC_models.ModelEvaluationData instance
        instance of `model` at `profile`
    success : bool
        whether model run was successful
    """

    required_datasets = ['output', 'predicted', 'observed', 'error']

    entry_name, site_name, pro_name = profile

    m = model(entry_name, site_name, pro_name, save_pkl=save_pkl)

    if force_rerun:
        m.purge_savedir(ask=False)

    if model is MENDData and tuple(profile) in MEND_excluded_profiles:
        logger.warning(f'Skipping {m}.')
        success = False

    elif all(f.exists() for f in m._file_groups['pickle'][required_datasets]):
        success = True

    elif m.forcing.isnull().values.sum():
        logger.error(f'NaN values in the forcing of {m}.')
        success = False

    else:
        logger.info(f'Running {m}.')
        try:
            for ds in required_datasets:
                if not m._file_groups['pickle'][ds].exists():
                    m.get(ds)
        except Exception as e:
            logger.exception(e)
            logger.error(f'Failed to run {m}.')
            success = False
        else:
            logger.success(f'Finished running {m}.')
            success = True

    return m, success


def _run_model_for_multiprocessing(args):
    model, profile, kwargs = args
    run_model(model, profile, **kwargs)


def run_all_models_all_profiles(njobs, models=None, profiles=None, **kwargs):
    """Run all models for all profiles on `njobs` CPU cores."""

    if models is None:
        models = get_all_models()
    if profiles is None:
        profiles = get_all_profiles()

    kwargs.setdefault('save_pkl', True) # save model outputs as pickle files

    if njobs == 1: # run on 1 core
        for model in models:
            for profile in profiles:
                run_model(model, profile, **kwargs)
        return

    # Make sure all local forcing data are ready
    for profile in profiles:
        forc = ForcingData(*profile, save_pkl=True)
        for dataset in ['constant', 'dynamic']:
            if not forc._file_groups['pickle'][dataset].exists():
                forc[dataset]

    # Run models in parallel on `njobs` cores
    list_of_args = [(m, p, kwargs) for p in profiles for m in models]
    with multiprocessing.Pool(njobs) as pool:
        pool.map(_run_model_for_multiprocessing, list_of_args)



def get_results(model, profile, **run_kwargs):
    m, success = run_model(model, profile, **run_kwargs)
    if success:
        predicted = m.predicted
        error = m.error
        if model is MENDData and profile in MEND_C_works_but_14C_fails:
            masked_columns = [c for c in error.columns if '14c' in c]
            predicted[masked_columns] = np.nan
            error[masked_columns] = np.nan
    else:
        # Create dataframes of the right shape and dtype, but filled with NaN
        columns = m.observed.columns
        index = m.observed.index
        predicted = pd.DataFrame(columns=columns, index=index, dtype='float')
        error = pd.DataFrame(columns=columns, index=index, dtype='float')
    return predicted, error


def get_results_all_profiles(model, profiles=None, **run_kwargs):
    if profiles is None:
        profiles = get_all_profiles()
    results = {
        tuple(profile): get_results(model, profile, **run_kwargs)
        for profile in profiles
    }
    predicted = pd.concat({
        profile: predicted for profile, (predicted, error) in results.items()
    })
    error = pd.concat({
        profile: error for profile, (predicted, error) in results.items()
    })
    predicted.index.names = ['entry_name', 'site_name', 'pro_name', 'date']
    error.index.names = ['entry_name', 'site_name', 'pro_name', 'date']
    return predicted, error


def get_all_results(models=None, profiles=None, **run_kwargs):
    if models is None:
        models = get_all_models()
    predicted, error = {}, {}
    for model in models:
        name = model.model_name
        predicted[name], error[name] = \
            get_results_all_profiles(model, profiles, **run_kwargs)
    return predicted, error


def get_bias_and_rmse(error=None, **run_kwargs):
    if error is None:
        error = get_all_results(**run_kwargs)[1]
    columns, index = SORTED_MODEL_NAMES, SORTED_VARIABLE_NAMES
    bias = pd.DataFrame(columns=columns, index=index, dtype='float')
    rmse = pd.DataFrame(columns=columns, index=index, dtype='float')
    for model_name, err in error.items():
        bias[model_name] = err.mean(axis=0)
        rmse[model_name] = np.sqrt((err * err).mean(axis=0))
    bias['average'] = bias.mean(axis=1)
    rmse['average'] = rmse.mean(axis=1)
    return bias, rmse
