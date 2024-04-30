import numpy as np
import rank_hist.rank_hist as rnk
import matplotlib.pyplot as plt
import importlib

from data_assimilation.data_assimilation_utilities import data_assimilation_workspace
# # Begin import exampledataL8.csv. This is data from an AR1 model with AR-parameter = 0.9 and noise std = 1.
# # Column 1: Verification
# # Column 2 to 6: Ensemble
# # Column 7: Strata
# # There are 5 ensemble members, the lead time is 8, and there are 3 strata.
# # There is also exampledataL4.csv with lead time 4 and 2 strata.
# joint_data = np.loadtxt('exampledataL4.csv')
# # End import exampledataL8.csv.

# Begin import Wei's data

# get UQ data
# get verifcation data

dfwspace = data_assimilation_workspace("/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin")

_max_step=300

data_ens = []
data_ver = []
data_ver_from_file = np.load(dfwspace.output_name("obs_data_reduced.npz", "cPDESolution"))

for t in range(1,_max_step+1):
    name = dfwspace.output_name("ensemble_at_obs_points_at_obs_t_{}.npy".format(t), "UQ") 
    data_ens.append(np.load(name))
    #print(name, ds.shape)
    npz_fname = 'obs_data_{}'.format(t)
    data_ver.append(data_ver_from_file[npz_fname])

#print(data_ver, len(data_ver))

data_ens = np.asarray(data_ens)
data_ver = np.asarray(data_ver)

print(data_ens.shape, data_ver.shape)


if 1:
    #ens_wei = np.genfromtxt("jochen_rankhist_ens_ux.csv", delimiter=",")  # ensemble data: shape (#of data, num_particles)
    #ver_wei = np.genfromtxt("jochen_rankhist_ver_ux.csv", delimiter=",")  # verification data: shape (# of data)

    ens_wei = data_ens[:, 0, :]
    ver_wei = data_ver[:, 0]
    ver_wei = ver_wei.reshape([ver_wei.shape[0], 1])
    joint_data = np.concatenate((ver_wei, ens_wei, ver_wei), axis = 1)
    # concatenating ver_wei at the end is a dummy that is going to be replaced with strat
    # End import Wei's data

    lead_time = 1 # Change this dep on data set
    nr_contrasts = 2 # Use 2 contrasts.
    # Too many contrasts and the test gets problems. Remember that the test has to estimate (nr_contrasts * nr_strata)^2 * (lead_time - 1) quantities!

    s_joint_data = joint_data.shape

    # Recompute stratification if desired. 
    nr_uq_strata = 1 # Change this number for your desired number of strata
    mn = np.median(joint_data[:, 0:s_joint_data[1]-1], axis = 1)
    ind = np.argsort(mn.reshape([s_joint_data[0], 1]), axis = 0)
    allranks = np.argsort(ind, axis = 0)
    strat = np.floor((nr_uq_strata * allranks) / s_joint_data[0])
    joint_data[:, s_joint_data[1]-1] = strat.reshape((s_joint_data[0],))

    # Normally no need to change anything below this line

    # Allocate verification, ensemble, and strata
    ver = joint_data[:, 0]
    ver = ver.reshape([ver.size, 1])
    ens = joint_data[:, 1:s_joint_data[1]-1]
    strat = joint_data[:, s_joint_data[1]-1]
    strat = strat.reshape([strat.size, 1])

    uq_strata = np.unique(strat)
    nr_uq_strata = uq_strata.size

    importlib.reload(rnk)

    pval, rnks_vals, covar_est, rnks_counts = rnk.rank_test(ver, ens, lead_time, strat, contrasts=nr_contrasts, return_counts = True)
    # Check rnk.rank_test? for explanation of input and output arguments.

    fig = plt.figure(100)
    rnks_array = np.array(rnks_counts)
    enslabels = np.array(rnks_counts.axes[0])
    for l in range(0, nr_uq_strata):
        ax = fig.add_subplot(nr_uq_strata, 1, l+1)
        ax.bar(enslabels, rnks_array[:, l])

    #fig = plt.figure(200)
    #plt.pcolor(covar_est.sum(axis = 2), cmap="Blues")
    #plt.colorbar()

    #fig = plt.figure(300)
    #plt.plot(np.arange(0, s_joint_data[0]), ens, 'bo')
    #plt.plot(np.arange(0, s_joint_data[0]), ver, 'ro')
    #plt.show(block=False)
    plt.savefig(dfwspace.output_name("hist.png", "UQ"))
    plt.close()
