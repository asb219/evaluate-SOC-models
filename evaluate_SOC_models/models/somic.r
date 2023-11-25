#!/usr/bin/env Rscript

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
