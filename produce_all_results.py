import argparse
import pandas as pd

from evaluate_SOC_models.data_manager import PandasExcelFile
from evaluate_SOC_models.observed_data import AllObservedData
from evaluate_SOC_models.results import (
    MEND_excluded_profiles,
    MEND_C_works_but_14C_fails,
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


TABLEPATH = SAVEPATH / 'tables'
PLOTPATH = SAVEPATH / 'plots'


if __name__ == '__main__': # necessary if multiprocessing

    models = (MIMICSData, MillennialData, SOMicData, CORPSEData, MENDData)
    profiles = AllObservedData().data.index

    ######################
    ### RUN ALL MODELS ###
    ######################

    parser = argparse.ArgumentParser(
        prog='produce_all_results',
        description='Evaluate all models, produce all result tables and plots',
        epilog=''
    )
    parser.add_argument('-njobs',
        help='number of CPU cores on which to run the models in parallel'
    )
    cmdline_arguments = parser.parse_args()
    njobs = cmdline_arguments.njobs
    njobs = 1 if njobs is None else int(njobs)

    if njobs > 1:
        run_all_models_all_profiles(njobs=njobs, models=models, profiles=profiles)


    #############################
    ### PRODUCE RESULT TABLES ###
    #############################

    predicted, error = get_all_results(models=models, profiles=profiles)
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

    def gCcm2_to_kgCm2(df):
        df = df.copy().rename(columns={'soc':'soc_kgCm2'})
        df['soc_kgCm2'] *= 10 # gC/cm2 -> kgC/m2
        return df

    excel_file = PandasExcelFile(TABLEPATH/'predicted.xlsx')
    for model_name, pred in predicted.items():
        excel_file.write(gCcm2_to_kgCm2(pred), sheet_name=model_name)

    excel_file = PandasExcelFile(TABLEPATH/'error.xlsx')
    for model_name, err in error.items():
        excel_file.write(gCcm2_to_kgCm2(err), sheet_name=model_name)

    gCcm2_to_kgCm2(observed).to_excel(TABLEPATH/'observed.xlsx')

    def gCcm2_to_kgCm2(df):
        df = df.copy().rename(index={'soc':'soc_kgCm2'})
        df.loc['soc_kgCm2'] *= 10 # gC/cm2 -> kgC/m2
        return df

    bias, rmse = get_bias_and_rmse(error=error)
    gCcm2_to_kgCm2(bias).to_csv(TABLEPATH/'all_bias.csv', float_format='%.1f')
    gCcm2_to_kgCm2(rmse).to_csv(TABLEPATH/'all_rmse.csv', float_format='%.1f')


    #####################
    ### PRODUCE PLOTS ###
    #####################

    plot_israd_map(
        show=False, save=PLOTPATH/'israd_map.pdf',
        save_kwargs=dict(bbox_inches='tight')
    )
    plot_israd_timeseries(
        figsize=(7,4),
        show=False, save=PLOTPATH/'israd_timeseries.pdf'
    )

    plot_boxplots_C(
        predicted=predicted_after_1995, observed=observed_after_1995,
        show=False, save=PLOTPATH/'results_boxplots_C.pdf'
    )
    plot_boxplots_14C(
        predicted=predicted_after_1995, observed=observed_after_1995,
        show=False, save=PLOTPATH/'results_boxplots_14C.pdf'
    )

    plot_predicted_vs_clay(
        predicted=predicted_after_1995, observed=observed_after_1995,
        show=False, save=PLOTPATH/'predicted_vs_clay.pdf',
        save_kwargs=dict(bbox_inches='tight')
    )
    plot_predicted_vs_temperature(
        predicted=predicted_after_1995, observed=observed_after_1995,
        show=False, save=PLOTPATH/'predicted_vs_temperature.pdf',
        save_kwargs=dict(bbox_inches='tight')
    )

    plot_predicted_vs_observed_all_models(
        predicted=predicted, observed=observed,
        show=False, save=PLOTPATH/'1-to-1_all.pdf'
    )
    for model in (MIMICSData, MillennialData, SOMicData, CORPSEData, MENDData):
        plot_predicted_vs_observed(
            model=model, predicted=predicted[model.model_name], observed=observed,
            show=False, save=PLOTPATH/f'1-to-1_{model.model_name}.pdf'
        )

    example_profile = ('Meyer_2012', 'Matsch', 'pasture')
    profile_name = '_'.join(example_profile)
    plot_predicted_14C_all_models(
        profile=example_profile,
        ylim=(-100, 500), t0='1950', t1='2015',
        show=False, save=PLOTPATH/f'predicted_14C_example_{profile_name}.pdf'
    )

    models = (MIMICSData, MillennialData, SOMicData, CORPSEData, MENDData)
    profiles = observed.index
    MEND_no_14C_profiles = MEND_excluded_profiles | MEND_C_works_but_14C_fails
    for model in models:
        savepath = PLOTPATH / 'predicted_14C_all' / model.model_name
        for profile in profiles:
            if model is MENDData and profile in MEND_no_14C_profiles:
                continue
            save = savepath / ('_'.join(profile) + '.pdf')
            plot_predicted_14C(model, profile, t0=1945, save=save, show=False)
