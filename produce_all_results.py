import argparse
import pandas as pd

from evaluate_SOC_models.observed_data import AllObservedData
from evaluate_SOC_models.results import (
    MEND_problematic_profiles,
    run_all_models_all_profiles,
    get_all_results,
    get_bias_and_rmse
)
from evaluate_SOC_models.models import (
    MIMICSData,
    MillennialData,
    SOMicData,
    CORPSEData,
    MENDData
)
from evaluate_SOC_models.path import SAVEPATH
from evaluate_SOC_models.plots import * # functions that start with `plot_`


if __name__ == '__main__': # necessary if multiprocessing

    ######################
    ### RUN ALL MODELS ###
    ######################

    parser = argparse.ArgumentParser(
        prog='produce_all_results',
        description='Evaluate all models, produce all result tables and plots',
        epilog=''
    )
    parser.add_argument('-njobs', help='number of cores to use for running models')
    cmdline_arguments = parser.parse_args()

    run_all_models_all_profiles(
        njobs=cmdline_arguments.njobs or 1, models=None, profiles=None
    )


    #############################
    ### PRODUCE RESULT TABLES ###
    #############################

    TABLEPATH = SAVEPATH / 'tables'
    TABLEPATH.mkdir(parents=True, exist_ok=True)

    predicted, error = get_all_results(models=None, profiles=None)
    observed = AllObservedData().data

    predicted_after_1995 = {
        model_name: pred[pred.index.get_level_values('date') > pd.to_datetime('1995')]
        for model_name, pred in predicted.items()
    }
    error_after_1995 = {
        model_name: err[err.index.get_level_values('date') > pd.to_datetime('1995')]
        for model_name, err in error.items()
    }
    observed_after_1995 = observed[observed.date > pd.to_datetime('1995')]

    for model_name, pred in predicted.items():
        pred.to_csv(TABLEPATH/f'predicted_{model_name}.csv')
    for model_name, err in error.items():
        err.to_csv(TABLEPATH/f'error_{model_name}.csv')
    observed.to_csv(TABLEPATH/'observed.csv')

    bias, rmse = get_bias_and_rmse(error=error)
    bias.to_csv(TABLEPATH/'all_bias.csv')
    rmse.to_csv(TABLEPATH/'all_rmse.csv')

    bias_after_1995, rmse_after_1995 = get_bias_and_rmse(error=error_after_1995)
    bias_after_1995.to_csv(TABLEPATH/'all_bias_after_1995.csv')
    rmse_after_1995.to_csv(TABLEPATH/'all_rmse_after_1995.csv')


    #####################
    ### PRODUCE PLOTS ###
    #####################

    PLOTPATH = SAVEPATH / 'plots'

    save_kwargs = dict(dpi=200)

    plot_israd_map(
        show=False, save=PLOTPATH/'israd_map.png',
        save_kwargs=dict(dpi=400, bbox_inches='tight')
    )
    plot_israd_timeseries(
        figsize=(7,4),
        show=False, save=PLOTPATH/'israd_timeseries.svg'
    )
    plot_israd_boxplot(
        show=False, save=PLOTPATH/'israd_boxplot.svg'
    )

    plot_boxplots_C(
        predicted=predicted_after_1995, observed=observed_after_1995,
        show=False, save=PLOTPATH/'results_boxplots_C.svg'
    )
    plot_boxplots_14C(
        predicted=predicted_after_1995, observed=observed_after_1995,
        show=False, save=PLOTPATH/'results_boxplots_14C.svg'
    )

    plot_predicted_vs_clay(
        predicted=predicted_after_1995, observed=observed_after_1995,
        show=False, save=PLOTPATH/'predicted_vs_clay.svg',
        save_kwargs=dict(bbox_inches='tight')
    )
    plot_predicted_vs_temperature(
        predicted=predicted_after_1995, observed=observed_after_1995,
        show=False, save=PLOTPATH/'predicted_vs_temperature.svg',
        save_kwargs=dict(bbox_inches='tight')
    )

    plot_predicted_vs_observed_all_models(
        predicted=predicted, observed=observed,
        show=False, save=PLOTPATH/'1-to-1_all.svg'
    )
    for model in (MIMICSData, MillennialData, SOMicData, CORPSEData, MENDData):
        plot_predicted_vs_observed(
            model=model, predicted=predicted[model.model_name], observed=observed,
            show=False, save=PLOTPATH/f'1-to-1_{model.model_name}.svg'
        )

    example_profile=('Meyer_2012', 'Matsch', 'pasture')
    profile_name = '_'.join(example_profile)
    plot_predicted_14C_all_models(
        profile=example_profile,
        ylim=(-100, 500), t0='1950', t1='2015',
        show=False, save=PLOTPATH/f'predicted_14C_example_{profile_name}.svg'
    )
    for model in (MIMICSData, MillennialData, SOMicData, CORPSEData, MENDData):
        savepath = PLOTPATH / 'predicted_14C_all' / model.model_name
        for profile in observed.index:
            if model is MENDData and profile in MEND_problematic_profiles:
                continue
            save = savepath / ('_'.join(profile) + '.svg')
            plot_predicted_14C(model, profile, save=save, show=False)
