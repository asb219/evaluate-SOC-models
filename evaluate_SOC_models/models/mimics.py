"""Based on "MIMICS-CN_forRelease.Rmd" at https://zenodo.org/records/3534562"""

import numpy as np
import pandas as pd
from numba import njit
import scipy.optimize

from evaluate_SOC_models.model_data import ModelEvaluationData


__all__ = ['MIMICSData']


def initialize_pools():
    """Copied and adapted from `initializePools` function"""

    LIT_1 = 1
    LIT_2 = 1
    MIC_1 = 1
    MIC_2 = 1
    SOM_1 = 1
    SOM_2 = 1
    SOM_3 = 1

    LIT_1_N = .1
    LIT_2_N = .1
    MIC_1_N = .1
    MIC_2_N = .1
    SOM_1_N = .1
    SOM_2_N = .1
    SOM_3_N = .1
    DIN     = .1

    LIT_1_14C = 1
    LIT_2_14C = 1
    MIC_1_14C = 1
    MIC_2_14C = 1
    SOM_1_14C = 1
    SOM_2_14C = 1
    SOM_3_14C = 1

    state = np.array((
        LIT_1, LIT_2,
        MIC_1, MIC_2,
        SOM_1, SOM_2, SOM_3,
        LIT_1_N, LIT_2_N,
        MIC_1_N, MIC_2_N,
        SOM_1_N, SOM_2_N, SOM_3_N,
        DIN,
        LIT_1_14C, LIT_2_14C,
        MIC_1_14C, MIC_2_14C,
        SOM_1_14C, SOM_2_14C, SOM_3_14C
    ))
    
    return state



def get_parameters(annual_NPP_C, clay_pct, Tsoil, input_lignin_pct, input_CN):
    """ All parameter values and comments are from `siteSpecificParameters`
    Comments with double-hash ## are lines of code removed by asb219
    """

    ##ANPP_C  = LTERdata$ANPP / 2       # convert to gC/m2/y from g/m2/y
    ANPP_C  = annual_NPP_C
    ##strSite = as.character(LTERdata$Site)  #convert site names to string
    ##nsites  = length(strSite)
    ##npts   = 6*10*14 #6 litter * 10 years * 14 sites
    clay  = clay_pct/100 ##LTERdata$CLAY2/100 
    tsoi  = Tsoil ##LTERdata$MAT
    ##nsites = length(LTERdata$Site)
    lig   = input_lignin_pct/100 #LTERdata$LIG/100
    Nnew  = 1/input_CN/2.5 #1/LTERdata$CN/2.5             #N in litter additions
    fMET1 = 0.85 - 0.013 * lig / Nnew    #as partitioned in Daycent

    #Parameters related to inputs
    EST_LIT_in  = ANPP_C / (365*24)   #gC/m2/h (from g/m2/y, Knapp et al. Science 2001)
    ##BAG_LIT_in  = 100      #gC/m2/h
    soilDepth       = 30  ## leave this as such!!
    ##h2y         = 24*365
    ##MICROtoECO  = soilDepth * 1e4 * 1e6 / 1e6   #mgC/cm3 to kgC/km2
    EST_LIT     = EST_LIT_in  * 1e3 / 1e4    #mgC/cm2/h 
    ##BAG_LIT     = BAG_LIT_in  * 1e3 / 1e4    #mgC/cm2/h
    fMET        = fMET1
    Inputs        = np.empty(2, np.float64)           #Litter inputs to MET/STR
    Inputs[0]     = (EST_LIT / soilDepth) * fMET      #partitioned to layers
    Inputs[1]     = (EST_LIT / soilDepth) * (1-fMET) ## units: mgC/cm3/h
    FI       = np.array((0.05,0.3))#c(0.05, 0.05)#

    ## BAG      = array(NA, dim=c(6,2))              #litter BAG inputs to MET/STR
    ## for (i in 1:6) {
    ## BAG[i,1]   = (BAG_LIT / soilDepth) * bagMET[i]      #partitioned to layers
    ## BAG[i,2]   = (BAG_LIT / soilDepth) * (1-bagMET[i])
    ## }

    #Parameters related to stabilization mechanisms
    fCLAY       = clay
    fPHYS    = 0.1 * np.array((.15 * np.exp(1.3*fCLAY), 0.1 * np.exp(0.8*fCLAY))) #Sulman et al. 2018
    fCHEM    = 3*np.array((0.1 * np.exp(-3*fMET) * 1, 0.3 * np.exp(-3*fMET) * 1)) #Sulman et al. 2018 #fraction to SOMc
    fAVAI    = 1-(fPHYS + fCHEM)
    desorb   = 2e-5      * np.exp(-4.5*(fCLAY)) #Sulman et al. 2018
    desorb   = 0.05*desorb
    Nleak   = 0.2#.1   #This is the proportion N lost from DIN pool at each hourly time step.

    #Parameters related to microbial physiology and pool stoichiometry
    CUE        = np.array((0.55, 0.25, 0.75, 0.35))  #for LITm and LITs entering MICr and MICK, respectively
    NUE        = .85         #Nitrogen stoichiometry of fixed pools
    CN_m        = 15
    CN_s        = (input_CN-CN_m*fMET)/(1-fMET)
    ##CN_s_BAG    =  (bagCN-CN_m*bagMET)/(1-bagMET)
    CN_r        =6#5
    CN_K        =10#8

    turnover      = np.array((5.2e-4*np.exp(0.3*(fMET)), 2.4e-4*np.exp(0.1*(fMET)))) #WORKS BETTER FOR N_RESPONSE RATIO
    ## turnover_MOD1 = np.sqrt(annual_NPP/100)  #basicaily standardize against NWT
    ## turnover_MOD1[turnover_MOD1 < 0.6] = 0.6 # correction not used in LIDET resutls 
    ## turnover_MOD1[turnover_MOD1 > 1.3] = 1.3      #Sulman et al. 2018
    turnover_MOD1 = min(1.3, max(0.6, np.sqrt(ANPP_C/100)))
    turnover      = turnover * turnover_MOD1
    turnover = turnover/2.2
    turnover = turnover**2*0.55/(.45*Inputs)
    densDep = 2#1 #This exponent controls the density dependence of microbial turnover. Currently anything other than 1 breaks things.

    fracNImportr  =  0 #No N import for r strategists
    fracNImportK  =  0.2 #Only K strategists can import N

    #Parameters related to temperature-sensitive enzyme kinetics
    TSOI        = tsoi
    #Calculate Vmax & (using parameters from German 2012)
    #from "gamma" simulations "best", uses max Vslope, min Kslope
    Vslope   = np.empty(6, np.float64)
    Vslope[0]= 0.043 #META LIT to MIC_1
    Vslope[1]= 0.043 #STRU LIT to MIC_1 
    Vslope[2]= 0.063 #AVAI SOM to MIC_1 
    Vslope[3]= 0.043 #META LIT to MIC_2 
    Vslope[4]= 0.063 #STRU LIT to MIC_2 
    Vslope[5]= 0.063 #AVAI SOM to MIC_2 
    Vint     = 5.47
    aV       = 8e-6
    aV       = aV*.06 #Forward
    Vmax     = np.exp(TSOI * Vslope + Vint) * aV

    Kslope   = np.empty(6, np.float64)
    Kslope[0]= 0.017 #META LIT to MIC_1
    Kslope[1]= 0.027 #STRU LIT to MIC_1 
    Kslope[2]= 0.017 #AVAI SOM to MIC_1 
    Kslope[3]= 0.017 #META LIT to MIC_2
    Kslope[4]= 0.027 #STRU LIT to MIC_2
    Kslope[5]= 0.017 #AVAI SOM to MIC_2
    Kint     = 3.19
    aK       = 10
    aK       = aK/20 #Forward
    Km       = np.exp(Kslope * TSOI + Kint) * aK

    #Enzyme kinetic modifiers:
    k        = 2.0    #2.0            #REDUCED FROM 3 TO 1, REDUCES TEXTURE EFFECTS ON SOMa decay
    a        = 2.0    #2.2            #increased from 4.0 to 4.5
    cMAX     = 1.4                    #ORIG 1.4 Maximum CHEM SOM scalar w/   0% Clay 
    cMIN     = 1.2                    #ORIG 1.4 Minimum CHEM SOM scalar w/ 100% Clay 
    cSLOPE   = cMIN - cMAX            #Slope of linear function of cSCALAR for CHEM SOM  
    pSCALAR  = a * np.exp(-k*(np.sqrt(fCLAY)))  #Scalar for texture effects on SOMp

    #------------!!MODIFIERS AS IN MIMICS2_b!!---------------
    MOD1     = np.array((10, 2*.75, 10, 3, 3*.75, 2))
    MOD2     = np.array(( 8, 2 ,4 * pSCALAR, 2, 4, 6 * pSCALAR))

    VMAX     = Vmax * MOD1
    KM       = Km / MOD2
    KO       = np.array((6,6))     #Values from Sulman et al. 2018

    return (
        Inputs, VMAX, KM, CUE,
        fPHYS, fCHEM, fAVAI, FI,
        turnover, ## LITmin, SOMmin, MICtrn,
        desorb, ## DEsorb, OXIDAT,
        ## LITminN, SOMminN, MICtrnN,
        ## DEsorbN, OXIDATN,
        KO,
        ## CNup, DINup, Nspill, Overflow,
        ## upMIC_1, upMIC_1_N,
        ## upMIC_2, upMIC_2_N,
        NUE, CN_m, CN_s, CN_r, CN_K, Nleak, densDep
    )



@njit
def FXEQ(state, Fm_input,
        Inputs, VMAX, KM, CUE, fPHYS, fCHEM, fAVAI, FI, turnover, desorb,
        KO, NUE, CN_m, CN_s, CN_r, CN_K, Nleak, densDep):
    """
    Most of the code is copied and adapted from the `FXEQ` function.
    All comments starting with a single # are from the original code.
    Comments starting with ## are by asb219.
    """

    (
        LIT_1, LIT_2,
        MIC_1, MIC_2,
        SOM_1, SOM_2, SOM_3,
        LIT_1_N, LIT_2_N,
        MIC_1_N, MIC_2_N,
        SOM_1_N, SOM_2_N, SOM_3_N,
        DIN,
        LIT_1_14C, LIT_2_14C,
        MIC_1_14C, MIC_2_14C,
        SOM_1_14C, SOM_2_14C, SOM_3_14C
    ) = state

    Inputs_1, Inputs_2 = Inputs
    VMAX_1, VMAX_2, VMAX_3, VMAX_4, VMAX_5, VMAX_6 = VMAX
    KM_1, KM_2, KM_3, KM_4, KM_5, KM_6 = KM
    CUE_1, CUE_2, CUE_3, CUE_4 = CUE
    fPHYS_1, fPHYS_2 = fPHYS
    fCHEM_1, fCHEM_2 = fCHEM
    fAVAI_1, fAVAI_2 = fAVAI
    FI_1, FI_2 = FI
    turnover_1, turnover_2 = turnover
    KO_1, KO_2 = KO


    #Carbon fluxes
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Flows to and from MIC_1
    LITmin_1 = MIC_1 * VMAX_1 * LIT_1 / (KM_1 + LIT_1)#LIT_1)   #MIC_1 decomp of MET lit
    LITmin_2 = MIC_1 * VMAX_2 * LIT_2 / (KM_2 + LIT_2)#LIT_2)   #MIC_1 decomp of STRUC lit
    SOMmin_1 = MIC_1 * VMAX_3 * SOM_3 / (KM_3 + SOM_3)#SOM_3)   #decomp of SOMa by MIC_1
    MICtrn_1 = MIC_1**densDep * turnover_1  * fPHYS_1 #MIC_1 turnover to PHYSICAL SOM
    MICtrn_2 = MIC_1**densDep * turnover_1  * fCHEM_1 #MIC_1 turnover to CHEMICAL SOM
    MICtrn_3 = MIC_1**densDep * turnover_1  * fAVAI_1 #MIC_1 turnover to AVAILABLE SOM

    #Flows to and from MIC_2
    LITmin_3 = MIC_2 * VMAX_4 * LIT_1 / (KM_4 + LIT_1)#LIT_1)   #decomp of MET litter
    LITmin_4 = MIC_2 * VMAX_5 * LIT_2 / (KM_5 + LIT_2)#LIT_2)   #decomp of SRUCTURAL litter
    SOMmin_2 = MIC_2 * VMAX_6 * SOM_3 / (KM_6 + SOM_3)#SOM_3)   #decomp of SOMa by MIC_2
    MICtrn_4 = MIC_2**densDep * turnover_2  * fPHYS_2                  #MIC_2 turnover to PHYSICAL  SOM
    MICtrn_5 = MIC_2**densDep * turnover_2  * fCHEM_2                  #MIC_2 turnover to CHEMICAL  SOM
    MICtrn_6 = MIC_2**densDep * turnover_2  * fAVAI_2                  #MIC_2 turnover to AVAILABLE SOM
    
    DEsorb    = SOM_1 * desorb  #* (MIC_1 + MIC_2)      #desorbtion of PHYS to AVAIL (function of fCLAY)
    OXIDAT    = ((MIC_2 * VMAX_5 * SOM_2 / (KO_2*KM_5 + SOM_2)) +#SOM_2)) +
                 (MIC_1 * VMAX_2 * SOM_2 / (KO_1*KM_2 + SOM_2)))#SOM_2)))  #oxidation of C to A

    #Nitrogen fluxes
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Flows to and from MIC_1
    LITminN_1 =  LITmin_1*LIT_1_N/(LIT_1+1e-100)#LITmin_1/CN_m
    LITminN_2 =  LITmin_2*LIT_2_N/(LIT_2++1e-100)#LITmin_2/CN_s
    SOMminN_1 =  SOMmin_1*(SOM_3_N/(SOM_3++1e-100))#SOMmin_1*(SOM_3_N/SOM_3)#*relRateN
    MICtrnN_1 =  MICtrn_1*MIC_1_N/(MIC_1+1e-100)#MICtrn_1/CN_r
    MICtrnN_2 =  MICtrn_2*MIC_1_N/(MIC_1+1e-100)#MICtrn_2/CN_r
    MICtrnN_3 =  MICtrn_3*MIC_1_N/(MIC_1+1e-100)#MICtrn_3/CN_r

    #Flows to and from MIC_2
    LITminN_3 =  LITmin_3*LIT_1_N/(LIT_1+1e-100)#LITmin_3/CN_m
    LITminN_4 =  LITmin_4*LIT_2_N/(LIT_2+1e-100)#LITmin_4/CN_s
    SOMminN_2 =  SOMmin_2*(SOM_3_N/(SOM_3+1e-100))#SOMmin_2*(SOM_3_N/SOM_3)#*relRateN
    MICtrnN_4 =  MICtrn_4*MIC_2_N/(MIC_2+1e-100)#MICtrn_4/CN_K
    MICtrnN_5 =  MICtrn_5*MIC_2_N/(MIC_2+1e-100)#MICtrn_5/CN_K
    MICtrnN_6 =  MICtrn_6*MIC_2_N/(MIC_2+1e-100)#MICtrn_6/CN_K
    
    DEsorbN    =  DEsorb*(SOM_1_N/(SOM_1+1e-100))#*relRateN
    OXIDATN    =  OXIDAT*(SOM_2_N/(SOM_2+1e-100))#*relRateN
    DINup_1   = (1-Nleak)*DIN*MIC_1/(MIC_1+MIC_2+1e-100) #Partitions DIN between microbial pools based on relative biomass
    DINup_2   = (1-Nleak)*DIN*MIC_2/(MIC_1+MIC_2+1e-100)

    #####
    upMIC_1    = CUE_1*(LITmin_1 + SOMmin_1) + CUE_2*(LITmin_2)
    upMIC_1_N  = NUE* (LITminN_1 + SOMminN_1) + NUE*(LITminN_2) + DINup_1
    CNup_1    = (upMIC_1)/(upMIC_1_N+1e-100) #Trying to optimize run speed here by avoiding /0
    Overflow_1 = (upMIC_1) - (upMIC_1_N)*min(CN_r, CNup_1)
    Nspill_1   = (upMIC_1_N) - (upMIC_1)/max(CN_r, CNup_1)
  
    upMIC_2    = CUE_3*(LITmin_3 + SOMmin_2) + CUE_4*(LITmin_4)
    upMIC_2_N  = NUE*(LITminN_3 + SOMminN_2) + NUE*(LITminN_4) + DINup_2
    CNup_2    = (upMIC_2)/(upMIC_2_N+1e-100)
    Overflow_2 = (upMIC_2) - (upMIC_2_N)*min(CN_K, CNup_2)
    Nspill_2   = (upMIC_2_N) - (upMIC_2)/max(CN_K, CNup_2)
    ######

    dLIT_1 = Inputs_1*(1-FI_1) - LITmin_1 - LITmin_3
    dLIT_2 = Inputs_2*(1-FI_2) - LITmin_2 - LITmin_4
    dMIC_1 = CUE_1*(LITmin_1 + SOMmin_1) + CUE_2*(LITmin_2) - (MICtrn_1 + MICtrn_2 + MICtrn_3) - Overflow_1
    dMIC_2 = CUE_3*(LITmin_3 + SOMmin_2) + CUE_4*(LITmin_4) - (MICtrn_4 + MICtrn_5 + MICtrn_6) - Overflow_2 
    dSOM_1 = Inputs_1*FI_1 + MICtrn_1 + MICtrn_4 - DEsorb 
    dSOM_2 = Inputs_2*FI_2 + MICtrn_2 + MICtrn_5 - OXIDAT
    dSOM_3 = MICtrn_3 + MICtrn_6 + DEsorb + OXIDAT - SOMmin_1 - SOMmin_2

    dLIT_1_N = Inputs_1*(1-FI_1)/CN_m - LITminN_1 - LITminN_3
    dLIT_2_N = Inputs_2*(1-FI_2)/CN_s - LITminN_2 - LITminN_4
    dMIC_1_N = NUE*(LITminN_1 + SOMminN_1) + NUE*(LITminN_2) - (MICtrnN_1 + MICtrnN_2 + MICtrnN_3) + DINup_1 - Nspill_1
    dMIC_2_N = NUE*(LITminN_3 + SOMminN_2) + NUE*(LITminN_4) - (MICtrnN_4 + MICtrnN_5 + MICtrnN_6) + DINup_2 - Nspill_2
    dSOM_1_N = Inputs_1*FI_1/CN_m + MICtrnN_1 + MICtrnN_4 - DEsorbN
    dSOM_2_N = Inputs_2*FI_2/CN_s + MICtrnN_2 + MICtrnN_5 - OXIDATN
    dSOM_3_N = MICtrnN_3 + MICtrnN_6 + DEsorbN + OXIDATN - SOMminN_1 - SOMminN_2

    dDIN = (
        (1-NUE)*(LITminN_1 + LITminN_2 + SOMminN_1) +  #Inputs from r decomp
        (1-NUE)*(LITminN_3 + LITminN_4 + SOMminN_2) +  #Inputs from K decomp
        Nspill_1 + Nspill_2 - DINup_1 - DINup_2    #Uptake to microbial pools and spillage
    )
    LeachingLoss = Nleak*DIN
    dDIN = dDIN-LeachingLoss #N leaching losses


    ####### The following 4 blocks of code were added by asb219 #######

    ## Fraction modern
    Fm_LIT_1 = LIT_1_14C / LIT_1
    Fm_LIT_2 = LIT_2_14C / LIT_2
    Fm_MIC_1 = MIC_1_14C / MIC_1
    Fm_MIC_2 = MIC_2_14C / MIC_2
    Fm_SOM_1 = SOM_1_14C / SOM_1
    Fm_SOM_2 = SOM_2_14C / SOM_2
    Fm_SOM_3 = SOM_3_14C / SOM_3

    ## 14C fluxes
    Inputs_1_14C = Inputs_1 * Fm_input
    Inputs_2_14C = Inputs_2 * Fm_input
    LITmin_1_14C, LITmin_3_14C = LITmin_1 * Fm_LIT_1, LITmin_3 * Fm_LIT_1
    LITmin_2_14C, LITmin_4_14C = LITmin_2 * Fm_LIT_2, LITmin_4 * Fm_LIT_2
    SOMmin_1_14C, SOMmin_2_14C = SOMmin_1 * Fm_SOM_3, SOMmin_2 * Fm_SOM_3
    MICtrn_1_14C, MICtrn_2_14C, MICtrn_3_14C = \
        MICtrn_1 * Fm_MIC_1, MICtrn_2 * Fm_MIC_1, MICtrn_3 * Fm_MIC_1
    MICtrn_4_14C, MICtrn_5_14C, MICtrn_6_14C = \
        MICtrn_4 * Fm_MIC_2, MICtrn_5 * Fm_MIC_2, MICtrn_6 * Fm_MIC_2
    Overflow_1_14C = Overflow_1 * Fm_MIC_1
    Overflow_2_14C = Overflow_2 * Fm_MIC_2
    DEsorb_14C = DEsorb * Fm_SOM_1
    OXIDAT_14C = OXIDAT * Fm_SOM_2

    ## Net 14C fluxes
    dLIT_1_14C = Inputs_1_14C*(1-FI_1) - LITmin_1_14C - LITmin_3_14C
    dLIT_2_14C = Inputs_2_14C*(1-FI_2) - LITmin_2_14C - LITmin_4_14C
    dMIC_1_14C = CUE_1*(LITmin_1_14C + SOMmin_1_14C) + CUE_2*(LITmin_2_14C) \
        - (MICtrn_1_14C + MICtrn_2_14C + MICtrn_3_14C) - Overflow_1_14C
    dMIC_2_14C = CUE_3*(LITmin_3_14C + SOMmin_2_14C) + CUE_4*(LITmin_4_14C) \
        - (MICtrn_4_14C + MICtrn_5_14C + MICtrn_6_14C) - Overflow_2_14C
    dSOM_1_14C = Inputs_1_14C*FI_1 + MICtrn_1_14C + MICtrn_4_14C - DEsorb_14C
    dSOM_2_14C = Inputs_2_14C*FI_2 + MICtrn_2_14C + MICtrn_5_14C - OXIDAT_14C
    dSOM_3_14C = MICtrn_3_14C + MICtrn_6_14C + DEsorb_14C + OXIDAT_14C - SOMmin_1_14C - SOMmin_2_14C
    
    ## Radioactive decay of 14C
    decay14C = np.log(2)/5730 / (365*24) # per hour
    dLIT_1_14C -= LIT_1_14C * decay14C
    dLIT_2_14C -= LIT_2_14C * decay14C
    dMIC_1_14C -= MIC_1_14C * decay14C
    dMIC_2_14C -= MIC_2_14C * decay14C
    dSOM_1_14C -= SOM_1_14C * decay14C
    dSOM_2_14C -= SOM_2_14C * decay14C
    dSOM_3_14C -= SOM_3_14C * decay14C
    ##########################################################


    return np.array([
        dLIT_1, dLIT_2, dMIC_1, dMIC_2, dSOM_1, dSOM_2, dSOM_3,
        dLIT_1_N, dLIT_2_N, dMIC_1_N, dMIC_2_N, dSOM_1_N, dSOM_2_N, dSOM_3_N, dDIN,
        dLIT_1_14C, dLIT_2_14C, dMIC_1_14C, dMIC_2_14C, dSOM_1_14C, dSOM_2_14C, dSOM_3_14C
    ])



@njit
def mimics_spinup(spinup_years, forc, NPP_mean,
        initial_state, Inputs,
        VMAX, KM, CUE, fPHYS, fCHEM, fAVAI, FI, turnover, desorb,
        KO, NUE, CN_m, CN_s, CN_r, CN_K, Nleak, densDep):
    state = initial_state.copy()
    for _ in range(spinup_years):
        for Fm_input, NPP in forc:
            Inputs_t = Inputs * NPP / NPP_mean
            state += FXEQ(state, Fm_input, Inputs_t,
                VMAX, KM, CUE, fPHYS, fCHEM, fAVAI, FI, turnover, desorb,
                KO, NUE, CN_m, CN_s, CN_r, CN_K, Nleak, densDep)
    return state


@njit
def mimics_realrun(forc, NPP_mean,
        initial_state, Inputs,
        VMAX, KM, CUE, fPHYS, fCHEM, fAVAI, FI, turnover, desorb,
        KO, NUE, CN_m, CN_s, CN_r, CN_K, Nleak, densDep):
    state = initial_state.copy()
    save = np.empty((len(forc), len(initial_state)), dtype=np.float64)
    save[0] = state
    for i, (Fm_input, NPP) in enumerate(forc[:-1]):
        Inputs_t = Inputs * NPP / NPP_mean
        state += FXEQ(state, Fm_input, Inputs_t,
            VMAX, KM, CUE, fPHYS, fCHEM, fAVAI, FI, turnover, desorb,
            KO, NUE, CN_m, CN_s, CN_r, CN_K, Nleak, densDep)
        save[i+1] = state
    return save



class MIMICSData(ModelEvaluationData):

    model_name = 'MIMICS'

    datasets = [
        'forcing', 'preindustrial_forcing', 'constant_forcing',
        'raw_output', 'output', 'predicted', 'observed', 'error'
    ]

    all_pools = [
        'LIT_1', # metabolic litter
        'LIT_2', # structural litter
        'MIC_1', # r-strategy microbes
        'MIC_2', # K-strategy microbes
        'SOM_1', # physically protected SOM
        'SOM_2', # chemically protected SOM
        'SOM_3', # available SOM
    ]
    fraction_to_pool_dict = {
        'LF': ['SOM_2'],
        'HF': ['SOM_1'],
        'bulk': ['MIC_1', 'MIC_2', 'SOM_1', 'SOM_2', 'SOM_3']
    }

    def __init__(self, entry_name, site_name, pro_name,
            spinup=4000, spinup_from_steady_state=200, # years
            *, save_pkl=True, save_csv=False, save_xlsx=False, **kwargs):

        super().__init__(entry_name, site_name, pro_name,
            save_pkl=save_pkl, save_csv=save_csv, save_xlsx=save_xlsx, **kwargs)

        self.spinup = spinup
        self.spinup_from_steady_state = spinup_from_steady_state


    def _process_forcing(self):
        forcing = self._forcing['dynamic'][
            ['Tsoil', 'Wsoil', 'NPP', 'Delta14Clit']
        ].copy()
        forcing['Tsoil'] -= 273.15 # Kelvin -> Celsius
        forcing['NPP'] *= 365*24*60*60 # gC/m2/s -> gC/m2/year
        forcing['Fin'] = forcing['Delta14Clit'] / 1000 + 1
        forcing = forcing.drop(columns=['Delta14Clit'])
        forcing = forcing.resample('H').ffill() # hourly time steps
        return forcing


    def _process_constant_forcing(self):
        cforc = super()._process_constant_forcing()
        cforc['CNlit_mean'] = self._forcing['dynamic']['CNlit'].mean()
        cforc['NPP_mean'] = self._forcing['dynamic']['NPP'].mean()
        cforc['NPP_mean'] *= 365*24*60*60 # gC/m2/s -> gC/m2/year
        cforc['Tsoil_mean'] = self._forcing['dynamic']['Tsoil'].mean()
        cforc['Tsoil_mean'] -= 273.15 # Kelvin -> Celsius

        # Lignin content of litter input in grassland
        # from Armstrong et al. (1950) DOI: 10.1017/S002185960004555X
        # Lignin content of litter input in shrubland and forests
        # from Rahman et al. (2013), DOI: 10.1080/02757540.2013.790380
        land_cover = cforc['pro_land_cover']
        if land_cover in ('rangeland/grassland', 'cultivated'):
            cforc['input_lignin_pct'] = 9.
        elif land_cover == 'shrubland':
            cforc['input_lignin_pct'] = 7.
        elif land_cover == 'forest':
            cforc['input_lignin_pct'] = 25.
        else:
            raise NotImplementedError(f'Run MIMICS on {land_cover} landscape')

        return cforc


    def _process_raw_output(self):

        cforc = self['constant_forcing']
        forcing = self['forcing']
        preindustrial_forcing = self['preindustrial_forcing']

        parameters = get_parameters(
            annual_NPP_C = cforc['NPP_mean'],
            clay_pct = cforc['clay'],
            Tsoil = cforc['Tsoil_mean'],
            input_lignin_pct = cforc['input_lignin_pct'],
            input_CN = cforc['CNlit_mean'],
        )

        (Inputs, VMAX, KM, CUE, fPHYS, fCHEM, fAVAI, FI, turnover, desorb,
        KO, NUE, CN_m, CN_s, CN_r, CN_K, Nleak, densDep) = parameters

        # First guess of steady state
        steady_state_C_N_guess = initialize_pools()[:-7]
        steady_state_F_guess = 0.95 + np.zeros(7, np.float64)

        # Find steady-state C and N stocks
        def func(state_C_N):
            state = np.append(state_C_N, [0]*7)
            return FXEQ(state, 1.0, *parameters)[:-7]
        steady_state_C_N, success_C_N = self._find_steady_state(
            func, steady_state_C_N_guess, bounds=(0, 1e7), name='C,N'
        )

        # Find steady-state fraction modern
        if success_C_N:
            steady_state_C = steady_state_C_N[:7]
            def func(F):
                state = np.append(steady_state_C_N, steady_state_C * F)
                deriv = FXEQ(state, 1.0, *parameters)[-7:]
                return deriv / steady_state_C
            steady_state_F, success_F = self._find_steady_state(
                func, steady_state_F_guess, bounds=(0.1, 1.1), name='F'
            )
        else:
            steady_state_F = None
            success_F = None

        # Spinup
        C_N = steady_state_C_N if success_C_N else steady_state_C_N_guess
        F = steady_state_F if success_F else steady_state_F_guess
        state = np.append(C_N, C_N[:7] * F)
        forc = preindustrial_forcing[['Fin', 'NPP']].values
        if success_C_N and success_F:
            spinup_years = self.spinup_from_steady_state
        else:
            spinup_years = self.spinup
        state = mimics_spinup(spinup_years, forc, cforc['NPP_mean'],
            state, Inputs,
            VMAX, KM, CUE, fPHYS, fCHEM, fAVAI, FI, turnover, desorb,
            KO, NUE, CN_m, CN_s, CN_r, CN_K, Nleak, densDep)

        # Real run
        forc = forcing[['Fin', 'NPP']].values
        save = mimics_realrun(forc, cforc['NPP_mean'],
            state, Inputs,
            VMAX, KM, CUE, fPHYS, fCHEM, fAVAI, FI, turnover, desorb,
            KO, NUE, CN_m, CN_s, CN_r, CN_K, Nleak, densDep)

        depth = cforc.lyr_bot - cforc.lyr_top
        save *= depth # mg/cm3 -> mg/cm2

        times = forcing.index
        columns = [
            'LIT_1', 'LIT_2', 'MIC_1', 'MIC_2', 'SOM_1', 'SOM_2', 'SOM_3',
            'LIT_1_N', 'LIT_2_N', 'MIC_1_N', 'MIC_2_N', 'SOM_1_N', 'SOM_2_N',
            'SOM_3_N', 'DIN',
            'LIT_1_14C', 'LIT_2_14C', 'MIC_1_14C', 'MIC_2_14C', 'SOM_1_14C',
            'SOM_2_14C', 'SOM_3_14C'
        ]
        save_df = pd.DataFrame(
            save, index=times, columns=columns
        ).resample('D').mean() # downsample to save on disk space and memory

        return save_df


    def _get_Cstocks_Delta14C(self, raw_output, fraction):
        cpools = self.fraction_to_pool_dict.get(fraction, [fraction])
        c14pools = [p + '_14C' for p in cpools]
        Cstocks = raw_output[cpools].sum(axis=1).values
        C14stocks = raw_output[c14pools].sum(axis=1).values
        Delta14C = (C14stocks / Cstocks - 1) * 1000
        Cstocks *= 1e-3 # mgC/cm2 -> gC/cm2
        return Cstocks, Delta14C
