from firedrake import *
import numpy as np
from data_assimilation.data_assimilation_utilities import data_assimilation_workspace
from data_assimilation.particle_filter import particle_filter_parameters as pfp 
from firedrake_utility import TorusMeshHierarchy
#from tqg.solver import TQGSolver
#import numpy as np
from stqg.solver import STQGSolver
from tqg.example2 import TQGExampleTwo as Example2params
from utility import Workspace
from scipy.linalg import ldl

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
    myvalue = - 0.5 * diff.T.dot(cov_inv).dot(diff) 
    
    return myvalue


def generate_observation(t, wspace):
    """
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


def compute_weight_exponents(t, wspace, cmesh, pfparams):
    obs = generate_observation(t, wspace)

    number_of_particles = 50
    weights_exponents = np.zeros(number_of_particles)

    cov_inv = np.load(wspace.get_parameter_files_dir() + '/obs_cov_sub_matrix_inv.npy')

    observation_points = pfparams.reduced_observation_mesh(t, wspace)
    
    for i in range(number_of_particles):
        filename = wspace.get_ensemble_members_dir() + '/ensemble_member_{}'.format(i)
        particle_ssh_values = observe_ssh(cmesh, filename, observation_points)
        #print(i, obs.shape, particle_ssh_values.shape)
        #print(filename)
        weights_exponents[i] = compute_log_likelihood(obs, particle_ssh_values, cov_inv)
        
    return weights_exponents


def compute_mse(ensemble_members, observations):
    """
    assumes dimension x ensemble_size for ensemble_members
    observations = dimension x 1
    """
    pass


def generate_bm_path(T,N):
    bm_inc = np.sqrt(T/N)*np.random.normal(0,1,N)
    bm     = np.concatenate(([0], np.cumsum(bm_inc)))
    return bm


if __name__ == "__main__":
    nx = 512
    cnx = 128 #64
    dt = 0.0001
    cdt = 0.0002  # not optimal but for the moment i don't care 
    num_steps = 5000
    dumpfreq = 10
    solve_flag = False
    write_visual = False 
    num_particles = 100 #50

    ensemble_size = ensemble_comm.size 
    ensemble_size = num_particles

    dfwspace = data_assimilation_workspace("/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin")

    pfparams = pfp()
    np.random.seed(10)

    T = 1. 
    
    # lead time 1
    def generate_ensemble_forecasts(_ltime, _initial_values, _num_fcasts=1):
        ### generate bm increment
        _fcast = np.sqrt(T / num_steps * _ltime) * np.random.normal(0, 1, (ensemble_size, _num_fcasts)) 
        return _initial_values + _fcast
        
    ltime = np.arange(1,11)

    truth     = generate_bm_path(T, num_steps)

    def generate_verification_data(_ltime):
        ensembles = generate_ensemble_forecasts(_ltime, truth[:-_ltime], num_steps -_ltime + 1)
        ens_means = np.mean(ensembles, axis=0)
        ens_sprds = np.mean((ensembles - ens_means)**2, axis=0)
        mses      = (ens_means - truth[_ltime:])**2
        
        return np.sqrt(np.mean(mses)), np.sqrt((ensemble_size + 1)/ensemble_size * np.mean(ens_sprds))

    rmses = []
    sprds = []
    
    for lt in ltime:
        rmse, sprd = generate_verification_data(lt)
        rmses.append(rmse)
        sprds.append(sprd)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1,1)
    ax.plot(ltime, rmses, c="blue", label="rmse")
    ax.plot(ltime, sprds, c="orange", label="sprd")
    ax.legend()
    ax.set_xlabel("lead times")
    
    plt.savefig(dfwspace.output_name("sprd_vs_rmse_test.png", "UQ"), dpi=300)
    plt.close()
