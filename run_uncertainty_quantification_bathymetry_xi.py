from firedrake import *
import numpy as np
from data_assimilation.data_assimilation_utilities import data_assimilation_workspace
from data_assimilation.particle_filter import particle_filter_parameters as pfp 
from firedrake_utility import TorusMeshHierarchy
#from tqg.solver import TQGSolver
#import numpy as np
from stqg.solver import STQGSolver
from tqg.example2 import TQGExampleTwo as Example2params
from scipy.linalg import ldl
from sys import argv

desired_spatial_rank = 1
ensemble = ensemble.Ensemble(COMM_WORLD, desired_spatial_rank)
spatial_comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm


def observe_ssh(cmesh, filename, observation_points):
    #cmesh = TorusMeshHierarchy(cnx, cny, 1., 1., 0, period="y", comm=spatial_comm).get_fine_mesh()
    cssh = Function(FunctionSpace(cmesh, "CG", 1))
    with DumbCheckpoint(filename, mode=FILE_READ) as chk:
        chk.load(cssh, name="SSH")
    values = np.asarray(cssh.at(observation_points, tolerance=1e-10))
    #print(values.shape)
    return values.reshape((len(values),1))


def ess_statistic(weights):
    """
    :param weights: normalised weights
    :return:
    """
    # return 1./ np.sum(np.square(weights))
    return 1. / np.dot(weights, weights)


def normalised_weights(_exponents, temp_increment=1.):
    """
    temp_increment < 1 means increased variance
    """
    z = np.asarray(_exponents)
    z -= np.max(z)
    weights = np.exp(z * temp_increment)
    return weights / np.sum(weights)


def compute_log_likelihood(x, y, cov_inv):
    """
    compute -0.5 * \sum \|x[i] - y[i]\|^2 / stddev^2
    for scalar vector entries

    :param x:
    :param y:
    :return:
    """
    #ndiff = (x - y)/self.obs_std_dev
    #myvalue = -0.5 * (np.dot(ndiff[:, 0], ndiff[:, 0]) + np.dot(ndiff[:, 1], ndiff[:, 1]))
    diff = x - y
    myvalue = - 0.5 * diff.T.dot(cov_inv).dot(diff) * 30
    
    return myvalue


def generate_observation(t, wspace):
    """
    returns truth + noise, and the actual truth value

    t here corresponds to saved file index,
    which is dumpfreq (=10) * dt (=0.0001)

    """
    # load covariance
    cov = np.load(wspace.get_parameter_files_dir() + '/obs_cov_sub_matrix.npy')
    L = np.linalg.cholesky(cov)
    z_ = np.random.normal(size=(len(L), 1))
    z  = L.dot(z_)

    truth = np.load(wspace.get_observation_data_dir() + '/obs_data_reduced.npz')['obs_data_{}'.format(t)]
    truth = truth.reshape((truth.shape[0], 1))
    #print(truth.files[:3])
    #print(truth.shape, z.shape)
    return truth + z, truth


def get_truth_full_grid(t, wspace):
    truth = np.load(wspace.get_observation_data_dir() + "/obs_data_full_grid.npz")['obs_data_full_grid_{}'.format(t)]
    truth = truth.reshape((truth.shape[0], 1))

    return truth


def compute_weight_exponents(t, wspace, cmesh, pfparams, number_of_particles):
    _obs,_truth = generate_observation(t, wspace)

    #number_of_particles = 50
    weights_exponents = np.zeros(number_of_particles)

    cov_inv = np.load(wspace.get_parameter_files_dir() + '/obs_cov_sub_matrix_inv.npy')

    observation_points = pfparams.reduced_observation_mesh(t, wspace)
    
    for i in range(number_of_particles):
        filename = wspace.get_ensemble_members_dir() + '/ensemble_member_{}'.format(i)
        particle_ssh_values = observe_ssh(cmesh, filename, observation_points)
        #print(i, _obs.shape, particle_ssh_values.shape)
        #print(filename)
        weights_exponents[i] = compute_log_likelihood(_obs, particle_ssh_values, cov_inv)
        
    return weights_exponents


def compute_mse(t, wspace, cmesh, pfparams, number_of_particles):
    """
    assumes dimension x ensemble_size for ensemble_members
    observations = dimension x 1
    """
    _obs, _truth = generate_observation(t, wspace)

    #number_of_particles = 50

    observation_points = pfparams.reduced_observation_mesh(t, wspace)

    ens_mean = np.zeros(_truth.shape)
    #print(ens_mean.shape)
    
    for i in range(number_of_particles):
        filename = wspace.get_ensemble_members_dir() + '/ensemble_member_{}'.format(i)
        particle_ssh_values = observe_ssh(cmesh, filename, observation_points)
        ens_mean += particle_ssh_values
        #print(i, _obs.shape, particle_ssh_values.shape)
        #print(filename)
    ens_mean /= number_of_particles

    return np.linalg.norm(ens_mean - _truth) / np.linalg.norm(_truth)
        

if __name__ == "__main__":
    nx = 512
    cnx = 32 #128 #64
    dt = 0.0001
    cdt = 0.0002  # not optimal but for the moment i don't care 
    num_steps = 100
    dumpfreq = 10
    dumpfreq_c = 5
    solve_flag = False
    write_visual = False 
    num_particles = 25 # was 50

    #batch_id = int(sys.argv[1])
    #print("batch_id: ", batch_id, ", ensemble_member_id: ",  ensemble_comm.rank + batch_id*ensemble_comm.size)

    ensemble_size = ensemble_comm.size 

    #mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y",comm=spatial_comm).get_fine_mesh()
    cmesh = TorusMeshHierarchy(cnx, cnx, 1., 1., 0, period="y", comm=spatial_comm).get_fine_mesh()
    #Vcg = FunctionSpace(cmesh, "CG", 1)
    
    # It doesn't matter what the actual params are, we just need a param object to pass to the Solver constructor
    #angry_dolphin_params_fmesh = Example2params(T, dt, mesh, bc='x', alpha=None)

    dfwspace = data_assimilation_workspace("/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin")
    dfwspace.ensemble_members_dir = dfwspace.sub_dir('EnsembleMembers')

    pfparams = pfp(dfwspace)
    #print(obs_mesh)
    #print(dfwspace.get_observation_data_dir())

    # initial ensemble ESS #############################
    _obs, _truth = generate_observation(0, dfwspace)
    weight_exponents = compute_weight_exponents(0, dfwspace, cmesh, pfparams, num_particles)
    #print(np.sqrt(compute_mse(0, dfwspace, cmesh, pfparams)))
    print('ess of initial ensemble: ', ess_statistic(normalised_weights(weight_exponents)))
    ####################################################

    np.random.seed(None)

    T = dt * dumpfreq   # every observation time -- 35 minutes

    if 1:

        angry_dolphin_params_cmesh = Example2params(T, cdt, cmesh, bc='x', alpha=None)
        spde_solver = STQGSolver(angry_dolphin_params_cmesh)

    if 1:
        # need to generate ensemble prediction, starting at t = t_n, to t_{n+1} = t_n + T
        # then compute rank with the observation at t_{n+1}

        t_next = 1 if len(argv) == 1 else int(argv[1])+1   # +1 -- assume the argv[1] = t_n
        truth = get_truth_full_grid(t_next, dfwspace)
        _obs, _truth = generate_observation(t_next, dfwspace)
        ensemble_values = np.zeros((len(truth), num_particles))
        ensemble_at_obs_values = np.zeros((len(_truth), num_particles))
        print("t_next ", t_next)

        for pid in range(num_particles):
            h5_data_name = dfwspace.output_name("ensemble_member_{}".format(pid), "EnsembleMembers")
            #print(h5_data_name) 
            spde_solver.load_initial_conditions_from_file(h5_data_name, comm=spatial_comm)
            data_output_name = dfwspace.get_particles_dir() + "/ensemble_member_{}".format(pid)
            # uses coarse pde data
            #spde_solver.solve(dumpfreq_c, "", data_output_name, ensemble, do_save_data=True, do_save_visual=False, do_save_spectrum=False, zetas_file_name=dfwspace.output_name("zetas.npy", "CalibrationData"))
            #spde_solver.solve(dumpfreq_c, "", data_output_name, ensemble, do_save_data=True, do_save_visual=False, do_save_spectrum=False,xi_scaling=0.001, bathymetry_xi=True) 
            spde_solver.solve(dumpfreq_c, "", data_output_name, ensemble, do_save_data=True, do_save_visual=False, do_save_spectrum=False, zetas_file_name=dfwspace.output_name("zetas.npy", "ParamFiles"))
            observation_points = pfparams.reduced_observation_mesh(t_next, dfwspace)
            _values_at_obs = np.asarray(spde_solver.ssh.at(observation_points, tolerance=1e-10))
            _values = spde_solver.ssh.dat.data[:]

            # uses alpha regularised pde data
            # need to save the data to common array
            ensemble_values[:, pid] += _values
            ensemble_at_obs_values[:, pid] += _values_at_obs
        np.save(dfwspace.output_name("ensemble_at_obs_points_at_obs_t_{}".format(t_next), "UQ"), np.asarray(ensemble_at_obs_values))
        np.save(dfwspace.output_name("ensemble_at_full_grid_at_obs_t_{}".format(t_next), "UQ"), np.asarray(ensemble_values))

    ##### single run, no reset, for mike cullen plot #####################
    if 0:
        ensembles = []
        ensembles_at_obs = []
        for t in range(num_steps):

            print(t+1, "/", num_steps)
            truth = get_truth_full_grid(t,dfwspace)
            _obs, _truth = generate_observation(t, dfwspace) # at obs sites
            
            ensemble_values = np.zeros((len(truth), num_particles)) # at full grid
            ensemble_at_obs_values = np.zeros((len(_truth), num_particles))

            for pid in range(num_particles):
                #dumpfreq_c = 5 #T / cdt 

                # solve particle to next time step
                h5_data_name = dfwspace.output_name("ensemble_member_{}".format(pid), "Particles") if t == 0 else dfwspace.output_name("ensemble_member_{}_1".format(pid), "Particles")
                print(h5_data_name)
                spde_solver.load_initial_conditions_from_file(h5_data_name, comm=spatial_comm)
                
                data_output_name = dfwspace.get_particles_dir() + "/ensemble_member_{}".format(pid)

                spde_solver.solve(dumpfreq_c,"", data_output_name, ensemble, do_save_data=True, do_save_visual=False, do_save_spectrum=False, zetas_file_name=dfwspace.output_name("zetas.npy", "ParamFiles"))
                #print(h5_data_name, norm(spde_solver.ssh))
                observation_points = pfparams.reduced_observation_mesh(t, dfwspace)
                _values_at_obs = np.asarray(spde_solver.ssh.at(observation_points, tolerance=1e-10))
                _values = spde_solver.ssh.dat.data[:]
                #print(_values.shape, ensemble_values[:, pid].shape)
                ensemble_values[:, pid] += _values
                ensemble_at_obs_values[:, pid] += _values_at_obs
            ensembles.append(ensemble_values)
            ensembles_at_obs.append(ensemble_at_obs_values)
        np.save(dfwspace.output_name("ensembles_at_obs_points", "UQ"), np.asarray(ensembles_at_obs))
        np.save(dfwspace.output_name("ensembles_at_full_grid", "UQ"), np.asarray(ensembles))
            
        # compute ensemble average

        # compute rmse

        # compute ensemble spread


    # plot rmse vs spread
    # this doesn't work for satellite data

    if 0:
        #ensembles = []
        #ensembles = np.load(dfwspace.output_name("ensembles_at_obs_points.npy", "UQ"))
        ensembles = np.load(dfwspace.output_name("ensembles_at_full_grid.npy", "UQ"))
        ensemble_size = ensembles.shape[-1]
        print(ensembles.shape, ensemble_size)
        #print(ensemble_members.shape)
        mses  = [] 
        sprds = []
        _means = []
        #num_steps =   
        #truth_all_steps = 
        truth_all_steps = np.load(dfwspace.get_observation_data_dir() + "/obs_data_full_grid.npz")
        for t in range(num_steps):
            # noisy_truth, truth = generate_observation(t,dfwspace)
            #truth = get_truth_full_grid(t,dfwspace)
            truth = truth_all_steps['obs_data_full_grid_{}'.format(t)]
            truth = truth.reshape((truth.shape[0], 1))
            #print(truth.shape, ensembles.shape)
            # observation_points = pfparams.reduced_observation_mesh(t, dfwspace)
            mean_t = np.mean(ensembles[t], axis=1)
            #print(mean_t.shape)
            _means.append(mean_t)

            mean_t = mean_t.reshape(len(mean_t), 1)
            #print(mean_t.shape, truth.shape)
            print(t+1, "/", num_steps) #, np.linalg.norm(mean_t - truth)**2)

            mse = np.mean((mean_t - truth[:, 0])**2)
            mses.append(mse)
            #mses.append(np.linalg.norm(mean_t - truth[:,0])**2)  # 2-norm

            #print(ensembles[t].shape, mean_t.shape)
            sprd = np.mean((ensembles[t] - mean_t)**2, axis=0)
            #print(sprd.shape)
            #sprd = np.mean(np.linalg.norm(ensembles[t] - mean_t, axis=0)**2) * ensemble_size / (ensemble_size - 1)
            sprd = np.sum(sprd) / (ensemble_size - 1)
            
            sprds.append(np.sqrt(sprd))
            
            #print(ensembles[t].T.shape, sprd.shape, mean_t.shape)

        mses = np.asarray(mses)
        _means = np.asarray(_means)
        sprds = np.asarray(sprds)
        print(_means.shape, mses.shape)
        print(sprds.shape)

        if 1:
            import matplotlib.pyplot as plt
            normalisations = np.arange(1, num_steps+1)
            fig, axes = plt.subplots(3, 1, figsize=(7, 8))

            axes[0].plot(normalisations, np.sqrt(mses), c="blue", label='rmse')
            axes[0].plot(normalisations, sprds, c="orange", label='sprd')
            axes[0].legend()

            axes[1].plot(normalisations, np.sqrt(mses), c="blue")
            axes[2].plot(normalisations, sprds, c="orange")

            fig.tight_layout()
            plt.savefig(dfwspace.output_name("rmse_sprd_full_grid.png", "UQ"), dpi=300)
            plt.close()

        
        
        

