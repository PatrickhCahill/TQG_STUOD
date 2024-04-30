###
###   Author: Wei Pan
###   Copyright   2017
###
###   utility.py
###
###   Contains utility functions and constant definitions
###

import os, errno
import numpy as np
import warnings

### shared ###############################################################################

pde_initial_truth_index = 730
spde_initial_truth_index = 274  # test case is 0, non test is 275
# obs_period = 0.1

delta_x = 1 / 4
n = 3


def remove_files(dirpath):
    import shutil
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)


def _gen_nonobs_grid():
    _temp_obs_pts = np.linspace(0, 1, n*n)
    _temp_nonobs_pts = [x for x in set(np.linspace(0, 1, 65)).difference(set(_temp_obs_pts))]
    nonobs_pts = [[x, y] for x in _temp_nonobs_pts for y in _temp_nonobs_pts]
    # now select a random subset of this set
    # random indices -- use combinations
    from random import sample
    rindices = sample(range(len(nonobs_pts)), n*n)
    return np.array([nonobs_pts[x] for x in rindices])


class GridUQ:
    """
    The 3 by 3 grid used for visualisation
    """

    # def __init__(self):
    gridpoints = np.array([[delta_x + i * delta_x, delta_x + j * delta_x] for j in range(n) for i in range(n)])
    # control = {'q': 0, 'ux': 1, 'psi': 2, 'uy': 3}
    # control = {'q': 0, 'ux': 1, 'psi': 2, 'uy': 3}
    # control_length = len(control)
    gridsize = len(gridpoints)
    gridpoints_nonobs =  _gen_nonobs_grid()
    gdxaxis = np.linspace(delta_x, n*delta_x, n)

    # 0[0.25 0.25]
    # 1[0.5  0.25]
    # 2[0.75 0.25]
    # 3[0.25 0.5]
    # 4[0.5 0.5]
    # 5[0.75 0.5]
    # 6[0.25 0.75]
    # 7[0.5 0.75]
    # 8[0.75 0.75]


def get_large_grid_indices(small_grid_xaxis, larger_grid_xaxis):
    """
    small_grid_xaxis = coarse grid
    larger_grid_xaxis = refined grid
    
    given a larger_grid x axis, find the indices of the larger_grid that contains the points in the small_grid

    e.g. np.linspace(0,1,65) would be the larger_rid_xaxis
    :return:
    """
    warnings.warn("depends on the specification of gridpoints, and the larger grid", FutureWarning)

    ix = np.isin(larger_grid_xaxis, small_grid_xaxis)
    # print(small_grid_xaxis, larger_grid_xaxis)
    # print(np.where(ix)[0])

    assert len(np.where(ix)[0]) > 0, "is not subgrid"

    index_list = []
    for i in range(len(small_grid_xaxis)):
        # print(list(np.where(ix)[0] + len(larger_grid_xaxis) * np.where(ix)[0][0] * (i + 1)))
        index_list += list(np.where(ix)[0] + len(larger_grid_xaxis) * np.where(ix)[0][0] * (i + 1))

    return index_list


def get_initial_truth_filename():
    return get_pde_filename(pde_initial_truth_index)


def get_spde_truth_initial_filename(_res):
    # return get_initial_ensemble_dir(_res) + 'particle_{}_0'.format(spde_initial_truth_index)
    return get_particle_initial_filename(_res, spde_initial_truth_index)


def get_particle_initial_filename(_res, _id):
    return get_initial_ensemble_dir(_res) + 'particle_{}_0'.format(_id)


def get_uq_particle_name(_id, _res=64, _var=0.5, suffix=''):
    """
    particle name when no particle filter
    :param _id:
    :param _res:
    :param _var:
    :return:
    """
    return output_directory('particle_{}{}'.format(_id, suffix), '/uq/SPDE/{}by{}/{}'.format(_res, _res, _var))


def get_particle_name(_id, _dir, suffix=''):
    return _dir + "particle_{}{}".format(_id, suffix)


def get_spde_truth_filename(_t, _dir):
    return "{}spde_truth_{}".format(_dir, _t)


def get_pde_truth_filename(_t, _dir):
    return "{}pde_truth_{}".format(_dir, _t)


def get_eof_dir(_res, _name=''):
    return output_directory(_name, '/eof/{}by{}'.format(_res, _res))


def get_eof_filename(_res, _var):
    return "{}zetas_{}.csv".format(get_eof_dir(_res), _var)


def get_pde_dir(_name=''):
    return output_directory(_name, '/PDEsolution')


def get_spde_truth_dir(_res=64, _var=0.5):
    return output_directory('', '/SPDETruths')


def get_pde_truth_dir():
    return output_directory('', '/PDETruths')


# def get_pf_twin_results_dir():
#     return output_directory('', '/uq/uqtests/TestResults/ParticleFilter/TwinExperiment')
#
#
# def get_pf_results_dir():
#     return output_directory('', '/uq/uqtests/TestResults/ParticleFilter')


def get_initial_ensemble_dir(part_msh_res=64, _name=''):
    return output_directory(_name, '/InitialEnsemble/{}by{}'.format(part_msh_res, part_msh_res))


def get_initial_ensemble_particlename(_id_, _res=64):
    warnings.warn('old', DeprecationWarning)
    # return utility.output_directory("particle_{}_0".format(job), _input_dir)
    return get_initial_ensemble_dir(_res, "particle_{}_0".format(_id_))


def get_filter_particle_dir():
    # posterior particles
    return output_directory('', '/ParticleFilter/Particles')


def get_resampled_particle_dir():
    return output_directory('','/ParticleFilter/ResampledParticles')

def get_prior_particle_dir():
    return output_directory('','/ParticleFilter/PriorParticles')

def get_observation_dir():
    return output_directory('', '/ParticleFilter/ObservationData')


def get_pde_filename(_num):
    return output_directory("q_" + str(_num), '/PDEsolution')


def get_uq_spde_results_dir(_res, _var):
    warnings.warn('old', DeprecationWarning)
    return output_directory('', '/uq/uqtests/SPDE/{}by{}/{}'.format(_res, _res, _var))


def get_uq_pde_results_dir(_res):
    warnings.warn('old', DeprecationWarning)
    return output_directory('', '/uq/uqtests/PDE/{}by{}'.format(_res, _res))


### utility functions ############################################################

def matrix_index_2d(i, j, size):
    return i*size + j


def mean_squared_error(x, y):
    """
    Assume both x and y are arrays of 2 dimensional data, we compute

    1/N * \sum_{i=1}{N} \| x[i] - y[i] \|_{l^2}^2

    where the norm inside the sum is the Euclidean 2 norm.
    :param x:
    :param y:
    :return:
    """
    diff = x - y
    value = np.dot(diff[:,0], diff[:,0]) + np.dot(diff[:,1], diff[:,1])
    return np.divide(value, len(diff), dtype=np.longdouble)


def mean_squared_value(x):
    """
    Assume x is an array of 2 dimensional data, we compute

    1/N * \sum_{i=1}{N} \| x[i] \|_{l^2}^2

    or 

    \sum_{i=1}{N} \| x[i] \|_{l^2}^2 * dx * dx
    
    dx = dy = 1/N   since our domain is [0,1]**2

    where the norm inside the sum is the Euclidean 2 norm.
    :param x:
    :return:
    """
    value = np.dot(x[:,0], x[:,0]) + np.dot(x[:,1], x[:,1])
    return np.divide(value, len(x), dtype=np.longdouble)


def matrix_index_3d(i, j, k, size2d):
    """
    compute the index in multiprocessing.Array for storing ft values
    equivalent to storing a 3d matrix (i,j,k)

    :param size2d: (i,j) = 0,...,size2d-1
    :param i: i axis
    :param j: j axis
    :param k: k axis
    :return: index for 1d array
    """
    return matrix_index_2d(i, j, size2d) + k * size2d*size2d


def output_directory(filename, additional_dir=''):
    dirname = os.getcwd() + additional_dir 
    # dirname = '/home/wpan1/Documents/Data/out' + additional_dir
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname, 0o755)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    return dirname + '/' + filename


def rescale_array(array, new_min, new_max):
    """ Rescale values in a array to the new range [min_, max_] """
    min_ = np.amin(array)
    max_ = np.amax(array)
    diff = new_max - new_min
    old_diff = max_ - min_
    # array[:] = np.array([ new_min + (r - min_) * diff / old_diff for r in array  ])
    array -= min_
    array *= diff / old_diff
    array += new_min


def kinetic_energy(velocities, delta_x):
    """ Compute 0.5 * sum norm(velocities) where velocities is the 
        serialised matrix of velocities on a square domain
    """
    # # l2 norm
    return 0.5 * sum([np.linalg.norm(v, ord = 2)**2 for v in velocities]) * delta_x * delta_x


def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]


def normalise(v):
    _norm = np.linalg.norm(v)
    if _norm < 1.e-10:
        return v
    return v / _norm


def create_grid(nx, ny, Lx=1., Ly=1.):
    x = np.arange(0, Lx + 1./nx, 1./nx)
    y = np.arange(0, Ly + 1./ny, 1./ny)

    return np.meshgrid(x, y)


def binary_search(lst, target):
    min = 0
    max = len(lst)-1
    avg = (min+max)/2
    # uncomment next line for traces
    # print lst, target, avg
    while (min < max):
        if (lst[avg] == target):
          return avg
        elif (lst[avg] < target):
          return avg + 1 + binary_search(lst[avg+1:], target)
        else:
          return binary_search(lst[:avg], target)

    # avg may be a partial offset so no need to print it here
    # print "The location of the number in the array is", avg
    return avg


def idiot_search(lst, target):
    index = 0
    while index < len(lst):
        if (lst[index] <= target) and (lst[index+1] > target):
            return index
        index += 1


def multithread_batch_length(total, nprocs):
    from math import floor
    if int(floor(total/nprocs)) > 0:
        return nprocs
    elif total % nprocs > 0:
        return total% nprocs
    else:
        return 0
