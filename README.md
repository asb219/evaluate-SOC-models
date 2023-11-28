# evaluate-SOC-models

Evaluate the performance of new-generation soil organic carbon models with radiocarbon (<sup>14</sup>C) data from [ISRaD](https://soilradiocarbon.org).

Evaluated models:
* Millennial v2 [^1]
* SOMic 1.0 [^2]
* MEND-new [^3]
* CORPSE-fire-response [^4]
* MIMICS-CN v1.0 [^5]


[^1]: Abramoff, R. Z., et al. (2022). Improved global-scale predictions of soil carbon stocks with Millennial Version 2.
_Soil Biology and Biochemistry, 164_, 108466. DOI: [10.1016/j.soilbio.2021.108466](https://doi.org/10.1016/j.soilbio.2021.108466).
Original source code in [rabramoff/Millennial](https://github.com/rabramoff/Millennial).

[^2]: Woolf, D., & Lehmann, J. (2019). Microbial models with minimal mineral protection can explain long-term soil organic carbon persistence.
_Scientific Reports, 9_(1), 6522. DOI: [10.1038/s41598-019-43026-8](https://doi.org/10.1038/s41598-019-43026-8).
Original source code in [domwoolf/somic1](https://github.com/domwoolf/somic1).

[^3]: Wang, G., et al. (2022). Soil enzymes as indicators of soil function: A step toward greater realism in microbial ecological modeling.
_Global Change Biology, 28_(5), 1935–1950. DOI: [10.1111/gcb.16036](https://doi.org/10.1111/gcb.16036).
Original source code in [wanggangsheng/MEND](https://github.com/wanggangsheng/MEND)

[^4]: This specific version of CORPSE is not published, but its source code is available on GitHub at [bsulman/CORPSE-fire-response](https://github.com/bsulman/CORPSE-fire-response).
The CORPSE model was first published in:
Sulman, B. N., et al. (2014). Microbe-driven turnover offsets mineral-mediated storage of soil carbon under elevated CO<sub>2</sub>.
_Nature Climate Change, 4_(12), 1099–1102. DOI: [10.1038/nclimate2436](https://doi.org/10.1038/nclimate2436).

[^5]: Kyker-Snowman, E., et al. (2020). Stoichiometrically coupled carbon and nitrogen cycling in the MIcrobial-MIneral Carbon Stabilization model version 1.0 (MIMICS-CN v1.0).
_Geoscientific Model Development, 13_(9), 4413–4434. DOI: [10.5194/gmd-13-4413-2020](https://doi.org/10.5194/gmd-13-4413-2020).
Original source code on Zenodo at [https://zenodo.org/records/3534562](https://zenodo.org/records/3534562).


## Repository contents

* `evaluate_SOC_models`: contains code to run the models and produce the results
* `produce_all_results.py`: script to run all models and produce plots and tables with results
* `MEND`: git submodule of [my fork](https://github.com/asb219/MEND) of MEND's original repository [wanggangsheng/MEND](https://github.com/wanggangsheng/MEND)
* `environment.yml`: specifies required python and R packages
* `config_defaults.ini`: default config file

[//]: # ( * `dump`: default directory for file storage )


## Getting started

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

If you don't have `conda`, download and install the newest version
of [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for a lightweight install.
If you prefer having a graphical user interface, get [Anaconda](https://www.anaconda.com/download/) instead.


### Install SOMic

[//]: # ( Make sure that the conda environment `eval14c` is activated. )

Install the SOMic model's R package into the conda environment with
```
Rscript -e "devtools::install_github('asb219/somic1@v1.1-asb219')"
```

This will install directly from [my fork](https://github.com/asb219/somic1)
of SOMic's original repository [domwoolf/somic1](https://github.com/domwoolf/somic1).



## Produce results


Coming soon



## [_Optional_] Customize configurations

For a list of configuration options, run
```
python -m evaluate_SOC_models.config --help
```

Changing any of the configurations will create a custom configuration file `config.ini`,
which takes precedence over `config_defaults.ini`.


### File storage location

The `evaluate_SOC_models` package produces around 26 GB of files
(downloads, model input, model output, plots),
which are stored in the `dump` directory by default.

You can check the full path of the file storage location with
```
python -m evaluate_SOC_models.config -get-dump
```

If you would like to store files in a different location, run the following command:
```
python -m evaluate_SOC_models.config -set-dump "/your/new/path/to/dump"
```
