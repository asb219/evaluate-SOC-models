#!/usr/bin/env Rscript

# Run SOMic (Woolf & Lehmann, 2019) and write model output to CSV file.
# This file is modified from "demo/SOMIC.r" (Woolf & Brunmayr, 2023), itself
# modified from "demo/SOMIC.r" (Woolf, 2024).
#
# * Modified source code of SOMic:
#     Woolf, D., & Brunmayr, A. S. (2023). "SOMic – GitHub fork asb219/somic1
#     (v1.1-asb219)". Zenodo. https://doi.org/10.5281/zenodo.11068749
#
# * Original source code of SOMic:
#     Woolf, D. (2024). "domwoolf/somic1: SOMic v 1.00 (v1.00)". Zenodo.
#     https://doi.org/10.5281/zenodo.10578048
#
# * Associated manuscript:
#     Woolf, D., & Lehmann, J. (2019). "Microbial models with minimal mineral
#     protection can explain long-term soil organic carbon persistence".
#     Scientific Reports, 9(1), 6522. https://doi.org/10.1038/s41598-019-43026-8
#
#
# Original work Copyright (C) 2023  D. Woolf & A. S. Brunmayr  (GPLv3 license)
#
# Modified work Copyright (C) 2024  Alexander S. Brunmayr  <asb219@ic.ac.uk>
#
# This file is part of the ``evaluate_SOC_models`` python package, subject to
# the GNU General Public License v3 (GPLv3). You should have received a copy
# of GPLv3 along with this file. If not, see <https://www.gnu.org/licenses/>.

cmdline_args <- commandArgs(trailingOnly=TRUE)

all_data_csvfile_path <- cmdline_args[1]
exp_const_csvfile_path <- cmdline_args[2]
output_csvfile_path <- cmdline_args[3]

library(SOMic)
library(data.table)

data("fit.par") # fitted parameters

all.data <- read.csv(all_data_csvfile_path) # time-dependent forcing data
exp.const <- read.csv(exp_const_csvfile_path) # constant forcing data

setDT(exp.const)
setDT(all.data)

somic.out = as.data.frame(
  somic(
    all.data,
    mic_vmax = fit.par['mic_vmax'],
    mic_km = fit.par['mic_km'],
    kdissolution = fit.par['kdissolution'],
    kdepoly = fit.par['kdepoly'],
    kdeath_and_exudates = fit.par['kdeath_and_exudates'],
    kdesorb = fit.par['kdesorb'],
    ksorb = fit.par['ksorb'],
    kmicrobial_uptake = fit.par['kmicrobial_uptake'],
    cue_0 = fit.par['cue_0'],
    mcue = fit.par['mcue'],
    mclay = fit.par['mclay'],
    clay = exp.const$clay,
    use_atsmd = 0,
    use_fraction_modern = exp.const$use_fraction_modern,
    init_spm_14c = exp.const$init_spm_14c,
    init_ipm_14c = exp.const$init_ipm_14c,
    init_doc_14c = exp.const$init_doc_14c,
    init_mb_14c = exp.const$init_mb_14c,
    init_mac_14c = exp.const$init_mac_14c,
  )
)

write.csv(somic.out, output_csvfile_path, row.names = FALSE)
