import argparse
import pandas as pd
import matplotlib.pyplot as plt

from evaluate_SOC_models.data_manager import PandasExcelFile
from evaluate_SOC_models.data_sources import Graven2017CompiledRecordsData
from evaluate_SOC_models.observed_data import AllObservedData, SelectedISRaDData
from evaluate_SOC_models.forcing_data import AllConstantForcingData
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
from evaluate_SOC_models.models.mimics2021 import MIMICS2021OutputFile
from evaluate_SOC_models.path import SAVEPATH
from evaluate_SOC_models.plots import * # functions that start with `plot_`


TABLEPATH = SAVEPATH / 'tables'
PLOTPATH = SAVEPATH / 'plots'


if __name__ == '__main__': # if-statement is necessary when multiprocessing

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

    excel_file = PandasExcelFile(TABLEPATH/'observed_and_predicted.xlsx')
    obs = observed.set_index('date', append=True)
    excel_file.write(gCcm2_to_kgCm2(obs), sheet_name='observed')
    for model_name, pred in predicted.items():
        excel_file.write(gCcm2_to_kgCm2(pred), sheet_name=model_name)

    excel_file = PandasExcelFile(TABLEPATH/'error.xlsx')
    for model_name, err in error.items():
        excel_file.write(gCcm2_to_kgCm2(err), sheet_name=model_name)

    excel_file = PandasExcelFile(TABLEPATH/'israd_and_soilgrids_data.xlsx')
    excel_file.write(gCcm2_to_kgCm2(SelectedISRaDData().data), sheet_name='ISRaD data')
    excel_file.write(AllConstantForcingData().data, sheet_name='forcing (ISRaD+SoilGrids)')

    def gCcm2_to_kgCm2(df):
        df = df.copy().rename(index={'soc':'soc_kgCm2'})
        df.loc['soc_kgCm2'] *= 10 # gC/cm2 -> kgC/m2
        return df

    bias, rmse = get_bias_and_rmse(error=error)
    gCcm2_to_kgCm2(bias).to_csv(TABLEPATH/'all_bias.csv', float_format='%.1f')
    gCcm2_to_kgCm2(rmse).to_csv(TABLEPATH/'all_rmse.csv', float_format='%.1f')


    ############################
    ### PRODUCE RESULT PLOTS ###
    ############################

    plot_israd_map(
        show=False, save=PLOTPATH/'israd_map.pdf',
        save_kwargs=dict(bbox_inches='tight')
    )
    plot_israd_timeseries(
        figsize=(7,4),
        show=False, save=PLOTPATH/'israd_timeseries.pdf'
    )

    plot_boxplots_C(
        predicted=predicted, observed=observed,
        show=False, save=PLOTPATH/'boxplots_C.pdf'
    )
    plot_boxplots_14C(
        predicted=predicted_after_1995, observed=observed_after_1995,
        show=False, save=PLOTPATH/'boxplots_14C.pdf'
    )

    for normalized_to_2000, path in [
        (True, PLOTPATH/'predicted_vs_environment_normalized_to_2000'),
        (False, PLOTPATH/'predicted_vs_environment_without_normalization')
    ]:
        plot_predicted_vs_clay(
            predicted=predicted_after_1995, observed=observed_after_1995,
            normalized_to_2000=normalized_to_2000,
            save=path/'predicted_vs_clay.pdf',
            save_kwargs=dict(bbox_inches='tight'), show=False
        )
        plot_predicted_vs_temperature(
            predicted=predicted_after_1995, observed=observed_after_1995,
            normalized_to_2000=normalized_to_2000,
            save=path/'predicted_vs_temperature.pdf',
            save_kwargs=dict(bbox_inches='tight'), show=False
        )
        for model in (MIMICSData, MillennialData, SOMicData, CORPSEData, MENDData):
            model_name = model.model_name
            predicted_model = {model_name: predicted_after_1995[model_name]}
            plot_predicted_vs_clay(
                predicted=predicted_model, observed=observed_after_1995,
                normalized_to_2000=normalized_to_2000,
                save=path/f'predicted_vs_clay_{model_name}.pdf',
                save_kwargs=dict(bbox_inches='tight'), show=False
            )
            plot_predicted_vs_temperature(
                predicted=predicted_model, observed=observed_after_1995,
                normalized_to_2000=normalized_to_2000,
                save=path/f'predicted_vs_temperature_{model_name}.pdf',
                save_kwargs=dict(bbox_inches='tight'), show=False
            )

    path = PLOTPATH/'predicted_vs_observed'
    for model in (MIMICSData, MillennialData, SOMicData, CORPSEData, MENDData):
        plot_predicted_vs_observed(
            model=model, predicted=predicted[model.model_name], observed=observed,
            show=False, save=path/f'{model.model_name}.pdf'
        )

    example_profile = ('Meyer_2012', 'Matsch', 'Pasture')
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


    #################################
    ### CHECK 14C IMPLEMENTATIONS ###
    #################################

    # Show that SOMic's original 14C implementation is inaccurate

    example_profile = ('Baisden_2002', 'Riverbank', 'Riverbank.1997')
    filename = 'check_14C_implementation_SOMic_' + '_'.join(example_profile)

    somic_good = SOMicData( # use the more accurate implementation with FM (default)
        *example_profile, use_fraction_modern=True
    ).output.loc['1950':, ['bulk_14c','LF_14c','HF_14c']]

    somic_bad = SOMicData( # use the inaccurate implementation with 14C ages
        *example_profile, use_fraction_modern=False, name='somic_bad'
    ).output.loc['1950':, ['bulk_14c','LF_14c','HF_14c']]

    excel_file = PandasExcelFile(TABLEPATH / (filename + '.xlsx'))
    excel_file.write(somic_good, sheet_name='using Fraction Modern (good)')
    excel_file.write(somic_bad, sheet_name='using 14C age (inaccurate)')

    atmosphere = Graven2017CompiledRecordsData().Delta14C.NH

    plt.figure(figsize=(6,5))
    plt.axhline(y=0, c='k', alpha=0.7, zorder=-10, lw=0.8)
    plt.plot(atmosphere, lw=3, c='k', label='atmospheric CO$_2$', zorder=0)
    plt.plot(somic_good['bulk_14c'], c='C0', label='bulk SOC', zorder=5, lw=2)
    plt.plot(somic_bad['bulk_14c'], c='C0', label='bulk SOC (inaccurate)', zorder=5, lw=2, alpha=0.5)
    plt.plot(somic_good['LF_14c'], c='C2', label='POM', zorder=2, lw=2)
    plt.plot(somic_bad['LF_14c'], c='C2', label='POM (inaccurate)', zorder=2, lw=2, alpha=0.5)
    plt.plot(somic_good['HF_14c'], c='C1', label='MAOM', zorder=1, lw=2)
    plt.plot(somic_bad['HF_14c'], c='C1', label='MAOM (inaccurate)', zorder=1, lw=2, alpha=0.5)
    plt.xlim((pd.to_datetime('1950'), pd.to_datetime('2020')))
    plt.xlabel('year', size=12)
    plt.ylabel('$\Delta^{14}$C (‰)', size=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(PLOTPATH / (filename + '.pdf'))
    plt.close()


    # Show that MIMICS's 14C implementation (Wang et al., 2021) doesn't work

    pools = MIMICS2021OutputFile.pools
    df14 = MIMICS2021OutputFile('14C').read().set_index('year').loc[1950:, pools]
    df12 = MIMICS2021OutputFile('12C').read().set_index('year').loc[1950:, pools]
    Delta14C = df14/df12 * 1000 - 1000

    Delta14C.to_excel(TABLEPATH / 'check_14C_implementation_MIMICS2021.xlsx', sheet_name='Delta14C')

    atmosphere = Graven2017CompiledRecordsData().Delta14C.loc['1950':, 'NH']
    atmosphere.index = atmosphere.index.year

    plt.figure(figsize=(6,5))
    plt.axhline(y=0, c='k', alpha=0.7, zorder=-10, lw=0.8)
    plt.plot(atmosphere, lw=3, c='k', label='atmospheric CO$_2$', zorder=0)
    for p in pools:
        plt.plot(Delta14C[p], label=p.replace('_','$_')+'$', lw=2)
    plt.xlim(1950, 2020)
    plt.xlabel('year', size=12)
    plt.ylabel('$\Delta^{14}$C (‰)', size=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(PLOTPATH / 'check_14C_implementation_MIMICS2021.pdf')
    plt.close()
