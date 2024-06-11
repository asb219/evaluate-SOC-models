"""
Run all models on all soil profiles, produce tables and plots of the results.

Copyright (C) 2024  Alexander S. Brunmayr  <asb219@ic.ac.uk>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import zipfile
import argparse
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt

from data_manager import PandasExcelFile

from evaluate_SOC_models.data import (
    Graven2017CompiledRecordsData,
    SelectedISRaDData,
    AllObservedData,
    AllConstantForcingData
)
from evaluate_SOC_models.results import (
    MEND_excluded_profiles,
    MEND_C_works_but_14C_fails,
    run_all_models_all_profiles,
    get_all_results,
    get_bias_and_rmse,
    SORTED_VARIABLE_NAMES
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


if __name__ == '__main__': # if-condition necessary when multiprocessing
    multiprocessing.freeze_support() # may be necessary on Windows (not tested)

    MODELS = (MIMICSData, MillennialData, SOMicData, CORPSEData, MENDData)
    PROFILES = AllObservedData().data.index
    MANUSCRIPT_LABEL = {} # labels of plots and tables in associated manuscript

    ######################
    ### RUN ALL MODELS ###
    ######################

    parser = argparse.ArgumentParser(
        prog='python -m produce_all_results',
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
        print(f'Running models in parallel with njobs={njobs}')
        run_all_models_all_profiles(njobs=njobs, models=MODELS, profiles=PROFILES)

    predicted, error = get_all_results(models=MODELS, profiles=PROFILES)
    observed = AllObservedData().data

    _1995 = pd.to_datetime('1995')
    predicted_after_1995 = {
        model_name: pred[pred.index.get_level_values('date') > _1995]
        for model_name, pred in predicted.items()
    }
    error_after_1995 = {
        model_name: err[err.index.get_level_values('date') > _1995]
        for model_name, err in error.items()
    }
    observed_after_1995 = observed[observed.date > _1995]


    #############################
    ### PRODUCE RESULT TABLES ###
    #############################

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

    israd_data = SelectedISRaDData().data
    constant_forcing = AllConstantForcingData().data
    excel_file = PandasExcelFile(TABLEPATH/'israd_and_soilgrids_data.xlsx')
    excel_file.write(gCcm2_to_kgCm2(israd_data), sheet_name='ISRaD data')
    excel_file.write(constant_forcing, sheet_name='forcing (ISRaD+SoilGrids)')

    def gCcm2_to_kgCm2(df):
        df = df.copy().rename(index={'soc':'soc_kgCm2'})
        df.loc['soc_kgCm2'] *= 10 # gC/cm2 -> kgC/m2
        return df

    bias, rmse = get_bias_and_rmse(error=error)
    gCcm2_to_kgCm2(bias).to_csv(TABLEPATH/'all_bias.csv', float_format='%.1f')
    gCcm2_to_kgCm2(rmse).to_csv(TABLEPATH/'all_rmse.csv', float_format='%.1f')
    MANUSCRIPT_LABEL[TABLEPATH/'all_bias.csv'] = 'Tab.2'
    MANUSCRIPT_LABEL[TABLEPATH/'all_rmse.csv'] = 'Tab.2'


    ############################
    ### PRODUCE RESULT PLOTS ###
    ############################

    plot_israd_map(
        show=False, save=PLOTPATH/'israd_map.pdf',
        save_kwargs=dict(bbox_inches='tight')
    )
    plot_israd_timeseries(
        figsize=(7,4), show=False, save=PLOTPATH/'israd_timeseries.pdf'
    )
    MANUSCRIPT_LABEL[PLOTPATH/'israd_map.pdf'] = 'Fig.1'
    MANUSCRIPT_LABEL[PLOTPATH/'israd_timeseries.pdf'] = 'Fig.2'

    plot_boxplots_C(
        predicted=predicted, observed=observed,
        show=False, save=PLOTPATH/'boxplots_C.pdf'
    )
    plot_boxplots_14C(
        predicted=predicted_after_1995, observed=observed_after_1995,
        show=False, save=PLOTPATH/'boxplots_14C.pdf'
    )
    MANUSCRIPT_LABEL[PLOTPATH/'boxplots_C.pdf'] = 'Fig.4'
    MANUSCRIPT_LABEL[PLOTPATH/'boxplots_14C.pdf'] = 'Fig.5'

    plot_data_vs_clay(
        plot_type='predicted', predicted=predicted_after_1995,
        observed=observed_after_1995, error=error_after_1995,
        normalized_to_2000=False,
        save=PLOTPATH/'predicted_vs_clay.pdf',
        save_kwargs=dict(bbox_inches='tight'), show=False
    )
    plot_data_vs_temperature(
        plot_type='predicted', predicted=predicted_after_1995,
        observed=observed_after_1995, error=error_after_1995,
        normalized_to_2000=False,
        save=PLOTPATH/'predicted_vs_temperature.pdf',
        save_kwargs=dict(bbox_inches='tight'), show=False
    )
    MANUSCRIPT_LABEL[PLOTPATH/'predicted_vs_clay.pdf'] = 'Fig.8'
    MANUSCRIPT_LABEL[PLOTPATH/'predicted_vs_temperature.pdf'] = 'Fig.7'

    for normalized_to_2000, path in [
        (False, PLOTPATH/'environment_plots'/'not_normalized'),
        (True, PLOTPATH/'environment_plots'/'normalized_to_year_2000')
    ]:
        for plot_type in ['predicted', 'error', 'absolute_error']:
            plot_data_vs_clay(
                plot_type=plot_type, predicted=predicted_after_1995,
                observed=observed_after_1995, error=error_after_1995,
                normalized_to_2000=normalized_to_2000,
                save=path/(plot_type+'_vs_clay.pdf'),
                save_kwargs=dict(bbox_inches='tight'), show=False
            )
            plot_data_vs_temperature(
                plot_type=plot_type, predicted=predicted_after_1995,
                observed=observed_after_1995, error=error_after_1995,
                normalized_to_2000=normalized_to_2000,
                save=path/(plot_type+'_vs_temperature.pdf'),
                save_kwargs=dict(bbox_inches='tight'), show=False
            )
            for model in MODELS:
                model_name = model.model_name
                plot_data_vs_clay(
                    plot_type=plot_type,
                    predicted={model_name: predicted_after_1995[model_name]},
                    observed=observed_after_1995,
                    error={model_name: error_after_1995[model_name]},
                    normalized_to_2000=normalized_to_2000,
                    save=path/f'{plot_type}_vs_clay_{model_name}.pdf',
                    save_kwargs=dict(bbox_inches='tight'), show=False
                )
                plot_data_vs_temperature(
                    plot_type=plot_type,
                    predicted={model_name: predicted_after_1995[model_name]},
                    observed=observed_after_1995,
                    error={model_name: error_after_1995[model_name]},
                    normalized_to_2000=normalized_to_2000,
                    save=path/f'{plot_type}_vs_temperature_{model_name}.pdf',
                    save_kwargs=dict(bbox_inches='tight'), show=False
                )

    path = PLOTPATH/'environment_plots'
    plot_sampling_date_vs_clay(
        show=False, save=path/'sampling_year_vs_clay.pdf'
    )
    plot_sampling_date_vs_temperature(
        show=False, save=path/'sampling_year_vs_temperature.pdf'
    )

    path = PLOTPATH/'predicted_vs_observed'
    for model in MODELS:
        plot_predicted_vs_observed_all_variables(
            model, predicted=predicted[model.model_name], observed=observed,
            show=False, save=path/f'all_variables_{model.model_name}.pdf'
        )
    for variable in SORTED_VARIABLE_NAMES:
        plot_predicted_vs_observed_all_models(
            variable, predicted=predicted, observed=observed,
            show=False, save=path/f'all_models_{variable}.pdf'
        )

    example_profile = ('Meyer_2012', 'Matsch', 'Pasture')
    filename = 'predicted_14C_example_' + '_'.join(example_profile) + '.pdf'
    plot_predicted_14C_all_models(
        profile=example_profile, ylim=(-100, 500), t0='1950', t1='2015',
        show=False, save=PLOTPATH/filename
    )
    MANUSCRIPT_LABEL[PLOTPATH/filename] = 'Fig.6'

    MEND_no_14C_profiles = MEND_excluded_profiles | MEND_C_works_but_14C_fails
    for model in MODELS:
        path = PLOTPATH / 'predicted_14C_all' / model.model_name
        for profile in PROFILES:
            if model is MENDData and profile in MEND_no_14C_profiles:
                continue
            save = path / ('_'.join(profile) + '.pdf')
            plot_predicted_14C(model, profile, t0=1945, save=save, show=False)


    #################################
    ### CHECK 14C IMPLEMENTATIONS ###
    #################################

    # Show that SOMic's original 14C implementation is inaccurate

    example_profile = ('Schrumpf_2013', 'Bugac', 'Bugac.1')
    filename = 'check_14C_implementation_SOMic_' + '_'.join(example_profile)

    somic_good = SOMicData( # use the more accurate implementation with FM
        *example_profile, use_fraction_modern=True # default is True
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
    plt.plot(somic_bad['bulk_14c'], c='C0', label='bulk SOC (inaccurate)', zorder=6, lw=2, alpha=0.5)
    plt.plot(somic_good['LF_14c'], c='C2', label='POM', zorder=3, lw=2)
    plt.plot(somic_bad['LF_14c'], c='C2', label='POM (inaccurate)', zorder=4, lw=2, alpha=0.5)
    plt.plot(somic_good['HF_14c'], c='C1', label='MAOM', zorder=1, lw=2)
    plt.plot(somic_bad['HF_14c'], c='C1', label='MAOM (inaccurate)', zorder=2, lw=2, alpha=0.5)
    plt.xlim((pd.to_datetime('1950'), pd.to_datetime('2020')))
    plt.xlabel('year', size=12)
    plt.ylabel('$\Delta^{14}$C (‰)', size=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    save = PLOTPATH / (filename + '.pdf')
    plt.savefig(save)
    plt.close()
    MANUSCRIPT_LABEL[save] = 'Fig.E1'


    # Show that MIMICS's 14C implementation (Wang et al., 2021) doesn't work

    filename = 'check_14C_implementation_MIMICS2021'

    pools = MIMICS2021OutputFile.pools
    df14 = MIMICS2021OutputFile('14C').read().set_index('year').loc[1950:, pools]
    df12 = MIMICS2021OutputFile('12C').read().set_index('year').loc[1950:, pools]
    Delta14C = df14/df12 * 1000 - 1000

    Delta14C.to_excel(TABLEPATH / (filename + '.xlsx'), sheet_name='Delta14C')

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
    save = PLOTPATH / (filename + '.pdf')
    plt.savefig(save)
    plt.close()
    MANUSCRIPT_LABEL[save] = 'Fig.E2'


    ###############################################################
    ### CREATE SUPPLEMENTARY MATERIAL FOR ASSOCIATED MANUSCRIPT ###
    ###############################################################

    # The quick and dirty way: Add all the files in TABLEPATH and PLOTPATH
    # to a compressed zip archive while labeling them as Tables and Figures

    def add_to_zipfile(zf, path, arc_dir='', label='', rename=''):

        label = MANUSCRIPT_LABEL.get(path, label)
        arc_filename = rename or (label+' '+path.name)
        arc_file = arc_dir + arc_filename

        if path.is_dir():
            arc_dir = arc_file + '/'
            zf.mkdir(str(arc_dir))
            added = [arc_dir]
            label_nr = 0
            for p in sorted(path.iterdir()):
                if p.name.startswith(('.')):
                    continue
                if p not in MANUSCRIPT_LABEL:
                    label_nr += 1
                added += add_to_zipfile(zf, p, arc_dir, label+f'.{label_nr}')
            return added

        zf.write(path, str(arc_file))
        return [arc_file]

    archive = SAVEPATH / 'Supplementary_Material_for_associated_manuscript.zip'

    with zipfile.ZipFile(archive, mode='w', compression=zipfile.ZIP_LZMA) as zf:
        added = add_to_zipfile(zf, TABLEPATH, '', 'Tab.S', rename='Tables')
        added += add_to_zipfile(zf, PLOTPATH, '', 'Fig.S', rename='Figures')
