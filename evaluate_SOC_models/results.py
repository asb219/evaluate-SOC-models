import multiprocessing
import numpy as np
import pandas as pd
from loguru import logger

from evaluate_SOC_models.observed_data import SelectedISRaDData
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


#ALL_MODELS = (MENDData, MillennialData, SOMicData, CORPSEData, MIMICSData)
SORTED_MODEL_NAMES = ['MEND', 'Millennial', 'SOMic', 'CORPSE', 'MIMICS']
SORTED_VARIABLE_NAMES = ['soc', 'bulk_14c', 'LF_14c', 'HF_14c', 'LF_c_perc', 'HF_c_perc']


def get_all_models():
    return (MENDData, MillennialData, SOMicData, CORPSEData, MIMICSData)

def get_all_profiles():
    return SelectedISRaDData().data.index


MEND_problematic_profiles = {
    ('Heckman_2018', 'HI_Andisol', 'WPL1204'), # still doesn't work
    ('Lemke_2006', 'Solling', 'Solling_DO_F1'), # still doesn't work
    ('Lemke_2006', 'Solling', 'Solling_DO_F2'), # still doesn't work
    ('Lemke_2006', 'Solling', 'Solling_DO_F3'), # still doesn't work
    ('McFarlane_2013', 'MA Harvard Forest', 'H4'), # still doesn't work
    ('McFarlane_2013', 'MI-Coarse UMBS', 'D4'), # now this doesn't work....
    ('McFarlane_2013', 'MI-Coarse UMBS', 'O2'), # now this doesn't work....
    ('McFarlane_2013', 'MI-Coarse UMBS', 'C7'), # still doesn't work
    #('McFarlane_2013', 'MI-Coarse UMBS', 'G3'), # WORKS NOW !!!
    ('Schrumpf_2013', 'Hesse', 'Hesse.1'), # still doesn't work
    #('Schrumpf_2013' 'Laqueuille' 'Laqueuille.1') # WORKS NOW !!!
}



def run_model(model, profile):
    """
    Parameters
    ----------
    model : :py:class:`evaluate_SOC_models.ModelEvaluationData` subclass
        class of model to run
    profile : tuple(str, str, str)
        3-tuple of ISRaD profile index (entry_name, site_name, pro_name)
    
    Returns
    -------
    m : :py:class:`evaluate_SOC_models.ModelEvaluationData` instance
        model 
    success : bool
        whether model was run successfully
    """

    entry_name, site_name, pro_name = profile

    m = model(entry_name, site_name, pro_name, save_pkl=True)

    if m._file_groups['pickle'].all_exist():
        success = True

    elif m.forcing.isnull().values.sum():
        logger.error(f'NaN values in the forcing of {m}.')
        success = False

    elif model is MENDData and tuple(profile) in MEND_problematic_profiles:
        logger.warning(f'Skipping {m}.')
        success = False

    else:
        logger.info(f'Running {m}.')
        try:
            m['error']
        except Exception as e:
            logger.exception(e)
            logger.error(f'Failed to run {m}.')
            success = False
        else:
            logger.success(f'Finished running {m}.')
            success = True

    return m, success


def _run_model_for_multiprocessing(args):
    run_model(*args)


def run_all_models_all_profiles(njobs, models=None, profiles=None):
    """Run all models for all profiles on `njobs` CPU cores."""

    if models is None:
        models = get_all_models()
    if profiles is None:
        profiles = get_all_profiles()

    # Make sure all local forcing data is ready
    for profile in profiles:
        forc = ForcingData(*profile, save_pkl=True)
        for dataset in ['constant', 'dynamic']:
            if not forc._file_groups['pickle'][dataset].exists():
                forc[dataset]

    if njobs == 1: # run on 1 core
        for model in models:
            for profile in profiles:
                run_model(model, profile)

    else: # run on multiple cores
        with multiprocessing.Pool(njobs) as pool:
            pool.map(_run_model_for_multiprocessing, [
                (model, profile) for model in models for profile in profiles
            ])



def get_results(model, profile):
    m, success = run_model(model, profile)
    if success:
        predicted = m.predicted
        error = m.error
    else:
        # Create dataframes of the right shape and dtype, but filled with NaN
        columns = m.observed.columns
        index = m.observed.index
        predicted = pd.DataFrame(columns=columns, index=index, dtype='float')
        error = pd.DataFrame(columns=columns, index=index, dtype='float')
    return predicted, error


def get_results_all_profiles(model, profiles=None):
    if profiles is None:
        profiles = get_all_profiles()
    all_results = {
        tuple(profile): get_results(model, profile) for profile in profiles
    }
    all_predicted = pd.concat({
        profile: predicted for profile,(predicted,error) in all_results.items()
    })
    all_error = pd.concat({
        profile: error for profile,(predicted,error) in all_results.items()
    })
    return all_predicted, all_error


def get_all_results(models=None, profiles=None):
    if models is None:
        models = get_all_models()
    predicted, error = {}, {}
    for model in models:
        name = model.model_name
        predicted[name], error[name] = get_results_all_profiles(model, profiles)
    return predicted, error


def get_bias_and_rmse(error=None):
    if error is None:
        error = get_all_results()[1]
    bias = pd.DataFrame(columns=SORTED_MODEL_NAMES, index=SORTED_VARIABLE_NAMES, dtype='float')
    rmse = pd.DataFrame(columns=SORTED_MODEL_NAMES, index=SORTED_VARIABLE_NAMES, dtype='float')
    for model_name, err in error.items():
        bias.loc[:, model_name] = err.mean(axis=0)
        rmse.loc[:, model_name] = np.sqrt((err * err).mean(axis=0))
    return bias, rmse
