# evaluate-SOC-models

Evaluate the performance of new-generation soil organic carbon (SOC) models with radiocarbon (<sup>14</sup>C) data of soil density fractions.

Evaluated models:

* Millennial v2 [^1]
* SOMic 1.0 [^2]
* MEND-new [^3]
* CORPSE-fire-response [^4]
* MIMICS-CN v1.0 [^5]


Topsoil data from [ISRaD](https://soilradiocarbon.org)[^6] used for model evaluation:

* <sup>14</sup>C content of bulk soil
* <sup>14</sup>C content of particulate organic matter (POM, "light" density fraction)
* <sup>14</sup>C content of mineral-associated organic matter (MAOM, "heavy" density fraction)
* SOC stocks
* Contribution of POM to SOC stocks
* Contribution of MAOM to SOC stocks



[^1]: Abramoff, R. Z., et al. (2022). Improved global-scale predictions of soil carbon stocks with Millennial Version 2.
_Soil Biology and Biochemistry, 164_, 108466. DOI: [10.1016/j.soilbio.2021.108466](https://doi.org/10.1016/j.soilbio.2021.108466).
Original source code in [rabramoff/Millennial](https://github.com/rabramoff/Millennial).

[^2]: Woolf, D., & Lehmann, J. (2019). Microbial models with minimal mineral protection can explain long-term soil organic carbon persistence.
_Scientific Reports, 9_(1), 6522. DOI: [10.1038/s41598-019-43026-8](https://doi.org/10.1038/s41598-019-43026-8).
Original source code in [domwoolf/somic1](https://github.com/domwoolf/somic1).

[^3]: Wang, G., et al. (2022). Soil enzymes as indicators of soil function: A step toward greater realism in microbial ecological modeling.
_Global Change Biology, 28_(5), 1935–1950. DOI: [10.1111/gcb.16036](https://doi.org/10.1111/gcb.16036).
Original source code in [wanggangsheng/MEND](https://github.com/wanggangsheng/MEND).

[^4]: This specific version of CORPSE is not published, but its source code is available on GitHub at [bsulman/CORPSE-fire-response](https://github.com/bsulman/CORPSE-fire-response).
The CORPSE model was first published in:
Sulman, B. N., et al. (2014). Microbe-driven turnover offsets mineral-mediated storage of soil carbon under elevated CO<sub>2</sub>.
_Nature Climate Change, 4_(12), 1099–1102. DOI: [10.1038/nclimate2436](https://doi.org/10.1038/nclimate2436).

[^5]: Kyker-Snowman, E., et al. (2020). Stoichiometrically coupled carbon and nitrogen cycling in the
MIcrobial-MIneral Carbon Stabilization model version 1.0 (MIMICS-CN v1.0).
_Geoscientific Model Development, 13_(9), 4413–4434. DOI: [10.5194/gmd-13-4413-2020](https://doi.org/10.5194/gmd-13-4413-2020).
Original source code on Zenodo at [https://zenodo.org/records/3534562](https://zenodo.org/records/3534562).

[^6]: ISRaD version 2.5.5.2023-09-20. Publication:
Lawrence, C. R., et al. (2020). An open-source database for the synthesis of soil radiocarbon data:
International Soil Radiocarbon Database (ISRaD) version 1.0.
_Earth System Science Data, 12_(1), 61–76. DOI: [10.5194/essd-12-61-2020](https://doi.org/10.5194/essd-12-61-2020).



## Repository contents

* `evaluate_SOC_models`: contains code to run the models and produce the results
* `produce_all_results.py`: script to run all models and produce plots and tables with the results
* `MEND`: git submodule of [my fork](https://github.com/asb219/MEND) of MEND's original repository [wanggangsheng/MEND](https://github.com/wanggangsheng/MEND)
* `environment.yml`: specifies required python and R packages
* `config_defaults.ini`: default config file

[//]: # ( * `dump`: default directory for file storage )



## Set up environment

### Clone this repository

Clone this repository, including its submodule `MEND`, with
```
git clone --recurse-submodules https://github.com/asb219/evaluate-SOC-models.git
```

Now the repository should be in a directory named `evaluate-SOC-models`.

Move into that directory with `cd evaluate-SOC-models`.


### Compile MEND

Move into the MEND submodule's directory and compile the Fortran code:
```
cd MEND
make
```

Then, move back to the main repository's directory with `cd ..`.


### Create conda environment

Create this project's python and R environment (named `eval14c` by default) with conda, and activate it:
```
conda env create -f environment.yml
conda activate eval14c
```

If you don't have conda, download and install the newest version
of [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for a lightweight install.
If you prefer having a graphical user interface, get [Anaconda](https://www.anaconda.com/download/) instead.


### Install SOMic

[//]: # ( Make sure that the conda environment `eval14c` is activated. )

Install the SOMic model's R package into the conda environment with
```
Rscript -e "devtools::install_github('asb219/somic1@v1.1-asb219')"
```

This will download and install directly from [my fork](https://github.com/asb219/somic1)
of SOMic's original repository [domwoolf/somic1](https://github.com/domwoolf/somic1).



## Produce results

Run the script `produce_all_results.py` to produce plots and tables
of the results:
```
python produce_all_results.py
```
On the first run, this will download 6.2 GB of forcing data
and run the 5 soil models on 77 selected topsoil profiles from ISRaD,
producing 6.3 GB of output data.
All resulting plots and tables are saved in the `dump/Results` folder.

To run the models in parallel, specify the number of CPU cores
with the `-njobs` flag.
For example, if you want to run on 7 cores, write
```
python produce_all_results.py -njobs 7
```

However, be aware that each run of the MEND model will produce
over 5 GB of temporary files that are loaded into RAM by python,
causing memory usage to spike up to 5 GB for about 1-2 second
before falling back down below 300 MB.
Make sure you have enough RAM and disk space to run MEND in parallel!


### Known issues with MEND

The MEND model experiences some numerical stability issues when run
with the forcing data of some of the 77 selected soil profiles.
I am currently preventing MEND to run on 12 blacklisted profiles which fail on my computer.

However, some issues with MEND seem to be specific to the compiler.
For example, with soil profile `('McFarlane_2013', 'MI-Coarse UMBS', 'G3')`,
MEND throws a SIGFPE when compiled with GNU Fortran 4.8.5 (Red Hat, Intel),
but runs without a problem when compiled with GNU Fortran 13.2.0 (MacOS, M1).

If MEND does not work for a specific soil profile on your computer,
add the profile to the set of `MEND_excluded_profiles` in `evaluate_SOC_models/results.py`.



## Customize configurations

For a list of configuration options, run
```
python -m evaluate_SOC_models.config --help
```

Changing any of the configurations will create a custom configuration file `config.ini`,
which takes precedence over `config_defaults.ini`.


### File storage location

Running the `produce_all_results.py` script will produce
12.5 GB of permanent files (downloads, model input and output, plots),
as well as a total of over 300 GB of temporary files which are written to disk
and quickly removed as the script runs MEND over the different soil profiles.
All those files are (permanently or temporarily) stored in the `dump` directory by default.

You can check the absolute path of the file storage location with
```
python -m evaluate_SOC_models.config -get-dump
```

If you would like to store the files in a different location, run the following command:
```
python -m evaluate_SOC_models.config -set-dump "/your/new/path/to/dump"
```


## Raising issues

If you encounter a problem with my code (or some other aspect of this project),
raise an [issue](https://github.com/asb219/evaluate-SOC-models/issues) on GitHub.
