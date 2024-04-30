#
#
#   Takes a cloud of particles and plots the uncertainty quantification at grid points
#
#
import matplotlib

matplotlib.use('Agg')

from data_assimilation.particle_filter import *
from data_assimilation.data_assimilation_utilities import data_assimilation_workspace
from data_assimilation.filterdiagnostics import FilterDiagnostics


class PlotError(Exception):
    pass


def run_spde_ensemble_trajectories():


    # obsscaling = .6
    df_wkspace = data_assimilation_workspace("/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin")

    df_wkspace.observation_data_dir = df_wkspace.sub_dir('cPDESolution-random-bathymetry')
    df_wkspace.zetas_dir = df_wkspace.sub_dir('CalibrationData-random-bathymetry')

    print(df_wkspace.observation_data_dir, df_wkspace.zetas_dir)

    pf_params = particle_filter_parameters(df_wkspace)
    pf = particle_filter(pf_params)

    pf.use_fine_res_obs_flag = True
    ##pf.uq_pf_flag = pf_flag
    ## visualdata = VisPFData(_res)

    # num_da_steps = 25 #int(round(time_interval / obs_period))    # do 25 d.a. steps
    num_da_steps = 150 #150 
    pf.run_particle_filter_uq_stability(_numsteps=num_da_steps, nproc=25)
    # diagnostics = FilterDiagnostics(outputdir=df_wkspace.diagnostic_dir)
    # diagnostics.generate_diagnostics_plots(num_da_steps, df_wkspace.get_diagnostic_dir(), df_wkspace.get_test_files_dir())

    ## visualdata.save_to_file_as_one_d(statisticsdir, 'weights', suffix)
    ## for name in ['mse', 'ess', 'mse_loc', 'mse_uq', 'ess_uq', 'mse_uq_loc', 'truthsetux', 'truthsetux_obsnoise',
    ##              'ensemblemeanux','ensemblemean_uqux', 'daparticlesetux', 'daparticleset_uqux']:
    ##     visualdata.save_to_file_as_one_d(statisticsdir, name, suffix)

if __name__ == "__main__":
    run_spde_ensemble_trajectories()

