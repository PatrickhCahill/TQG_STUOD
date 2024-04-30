import numpy as np
import scipy.stats as stats
from pandas import crosstab

def rank_test(ver, ens, lead_time, strat, contrasts=[], return_counts = False):
    """function [pval, rnks_vals, covar_est] = rank_test(ver, ens, lead_time, strat, contrasts=[])    
    
    function [pval, rnks_vals, covar_est, rnks_counts] = rank_test(ver, ens, lead_time, strat, contrasts=[], return_counts = True)
    
    Performs a generalised GOF test for rank histograms from ensemble
    forecasts. 

    Input arguments:

    ver -- The verification, an array with dimensions [nr_tstamps, 1], first dimension representing time
    ens -- The ensemble, an array with dimensions [nr_tstamps, nr_ens] with first dimension representing time and second dimension representing ensemble members
    lead_time -- The lead time of the forecast.
    strat -- stratum, a column vector of dimension nr_tstamps-by-1. Entries to this vector should be from a set of S different symbols (the exact symbol is irrelevant). strat[n, 0] gives the stratum of the n'th sample, and samples with the same stratum will be associated with the same histogram (out of S histograms). A warning is given if S > log(nr_tstams) or if the number of samples in the different strata differs by more than 20%.

    Optional arguments
    
    contrasts=ctr -- If ctr=[] (default), a set of nr_ens orthonormal contrasts is computed. A contrast is a vector x so that sum(x) = 0, and there are nr_ens possible contrasts. The contrasts are generated using contrast_gen(nr_ens + 1, nr_ens). If ctr is a number m (not exceeding nr_ens), then only m contrasts are computed (by invoking contrast_gen(nr_ens + 1, m)). If ctr is a [m1, m2] shaped array, the columns are interpreted as contrasts, and the shape must be m1 = nr_ens + 1, m2 <= nr_ens.  

    Output arguments:

    pval -- The p-value of the test
    rnks_vals -- The ranks of the verification

    covar_est -- Estimator of the (matrix valued) covariance function of the histogram, projected onto the contrasts. This is a 3-dimensional array with dimensions [nr_ctr * S, nr_ctr * S, lead_time]. The entry covar_est[c1 + s1 * nr_contrasts, c2 + s2 * nr_contrasts, l] is the correlation between Rank(n) resp Rank(n + l), projected onto contrast and stratum (c1+1, s1+1), resp contrast and stratum (c2+1, s2+1). We may calculate C = np.sum(covar_est, axis = 2), then the entry C[c1 + s1 * nr_contrasts, c2 + s2 * nr_contrasts] is the correlation between the histograms for strata s1+1 and s2+1, both projected onto contrasts c1+1 and c2+1, respectively, and divided by sqrt(nr_tstamps).

    rnks_counts -- (if return_counts=True) The rank histogram bars in a data frame with dimensions [S, nr_contrasts]

    Disclaimer: Use at your own risk!

    (c) Jochen Broecker, 2018
    """
    # function body

    s_ver = ver.shape
    s_ens = ens.shape
    nr_tstamps = s_ver[0]
    nr_ens = s_ens[1]
    nr_ranks = 1 + nr_ens
    sqrt_ranks = np.sqrt(nr_ranks)
    nr_contrasts = nr_ens
    uq_strata, strat_inv, strata_counts = np.unique(strat, return_inverse = True, return_counts = True)
    strat_inv = strat_inv.reshape([nr_tstamps, 1])
    nr_uq_strata = uq_strata.size
    
    if (np.size(contrasts) == 0):
        contrasts = nr_ens
    if (type(contrasts) == type(0)):
        if (contrasts < nr_ranks):
            the_contrasts = contrast_gen(nr_ranks, contrasts)
            nr_contrasts = contrasts
        else: 
            print('Error: Number of contrasts must not be larger than number of ensemble members!')
    elif (type(contrasts) == type(np.array([]))):
        the_contrasts = contrasts
        s_the_contrasts = the_contrasts.shape
        if ((s_the_contrasts[0] != nr_ranks) or (s_the_contrasts[1] > nr_ens)):
            print('Error: Contrast matrix has wrong dimensions!')
        nr_contrasts = s_the_contrasts[1]
    else:
        print('Type of argument contrasts not understood')

    # sort out strata
    if (nr_uq_strata > np.log(nr_tstamps)):
        print('Warning: Number of strata should not be larger than log(nr_tstamps)')
    elif np.sqrt(strata_counts.std()/strata_counts.mean() > 0.4):
        print('Warning: Number of samples in strata varies by more than 40%')

    # Form scaled indicator variables
    ind = np.argsort(np.concatenate((ver, ens), axis=1), axis=1)
    allranks = np.argsort(ind, axis=1)
    rnks_vals = allranks[:,0] + 1
    Z = (ind == 0) * 1.0;
    Z = sqrt_ranks * Z
    contrasted_Z = Z @ the_contrasts

    # Bring in the strata (to do dispense with loop using np.repeat)
    new_Z = np.zeros([nr_tstamps, nr_uq_strata * nr_contrasts])
    for s in range(0, nr_uq_strata):
        new_Z[:, s * nr_contrasts : (s+1) * nr_contrasts] = contrasted_Z * np.tile((strat_inv == s), [1, nr_contrasts])
        new_Z[:, s * nr_contrasts : (s+1) * nr_contrasts] =         new_Z[:, s * nr_contrasts : (s+1) * nr_contrasts] / np.sqrt(strata_counts[s]/nr_tstamps)

    # perform generalised chi square test
    pval, covar_est = gen_chi_squ(new_Z, lead_time)

    # compute actual histograms for visual display
    if return_counts:
        rnks_counts = crosstab(rnks_vals.reshape([nr_tstamps,]),  strat_inv.reshape([nr_tstamps,]))
                                                            
        return(pval, rnks_vals, covar_est, rnks_counts)
    else:
        return(pval, rnks_vals, covar_est)

def gen_chi_squ(Z, corr_time):
    """function [pval, covar_func] = gen_chi_squ(Z, corr_time)
    Performs a generalised chi square test. 

    Input arguments:

    Z -- An array with dimensions [nr_tstamps, d]
    corr_time -- correlation time of Z

    Purpose:

    Suppose that Z satisfies a Central Limit Theorem, in the sense that 

    X = (1/sqrt(nr_tstamps)) sum(Z, axis = 0) 

    is asymptotically normal with mean zero and covariance matrix M, then 

    t = X * inv(M) * transpose(X)

    is chi-square distributed with d DOF. The covariance matrix M is in theory given by the sum over the (matrix valued) covariance function of Z (which is assumed to converge).  

    The code will estimate the covariance function and also M. It is assumed however that Z is mean free and has unit covariance. It is further assumed that Z has no correlations at a lag given by corr_time and beyond.

    Output arguments:

    pval -- The p-value of the test
    covar_func -- Estimator of the covariance function. This is an array with dimensions [d, d, corr_time]. There is some redundancy here since covar_func[:, :, 0] is the d-by-d unit matrix.


    Disclaimer: Use at your own risk!

    (c) Jochen Broecker, 2018
    """
    # function body

    s_Z = Z.shape
    nr_tstamps = s_Z[0]
    dof = s_Z[1]
    Z = Z / np.sqrt(nr_tstamps)
    
    # Prepare estimating variance
    covar_func = np.zeros([dof, dof, corr_time])
        
    # We start with correlation lag = 0
    covar_func[:, :, 0] = np.eye(dof)
    var_est = covar_func[:, :, 0]

    if (corr_time > 1):
        for l in range(1, corr_time):
            buffer_length = nr_tstamps - l
            dummy = np.transpose(Z[0:buffer_length, :]) @ Z[l:buffer_length+l, :]
            covar_func[:, :, l] = dummy + np.transpose(dummy)
            var_est = var_est + covar_func[:, :, l]

    inv_var_est = np.linalg.inv(var_est)
    d = np.sum(Z, axis=0)
    gofstat = (d @ inv_var_est) @ np.transpose(d)
    pval = 1 - stats.chi2.cdf(gofstat, dof)
                                                            
    return(pval, covar_func)


def contrast_gen(nr_ranks, nr_contrasts):
    """function contrasts = contrast_gen(nr_ranks, nr_contrasts)
    
    A reasonable set of orthonormal contrasts is computed. A contrast
    is a vector x so that sum(x) = 0.
  
    nr_ranks -- no. of ranks (= 1 + no. of ensemble members)
    nr_contrasts -- no. of contrasts returned. Must not be larger than no. of ensemble members

    return -- Array of dimension [nr_ranks, nr_contrasts] with columns representing orthonormal contrasts.

    Disclaimer: Use at your own risk!

    (c) Jochen Broecker, 2018
    """
    # function body

    V = np.zeros((nr_ranks, nr_contrasts + 1))
    V[:, 0] = np.ones((nr_ranks))
    ranks_recentered = np.linspace(1, nr_ranks, nr_ranks) - (nr_ranks + 1) / 2.0
    for ctr in range(1, nr_contrasts + 1):
        w = ranks_recentered ** ctr
        w = (w - np.mean(w)) / np.std(w)
        V[:, ctr] = w
    G = np.linalg.qr(V)
    return(G[0][:,1:])
