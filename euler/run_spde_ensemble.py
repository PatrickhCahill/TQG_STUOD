#
#   Author: Wei Pan
#   Copyright   2017
#
#   run_spde_ensemble.py
#
#   run spde from an intial ensemble set for a set number of eddy turnover times
#

import particle
from multiprocessing import Process
from utilitiesFiredrake import *


def wrapper(args):
    """
    wrapper function for particles, for multiprocessing.Process
    :param args:
    :return:
    """
    time_interval, dumpf, msh, _dt, _job, _eof_dir, _var_, _output_dir = args
    name = _output_dir + 'particle_' + str(_job)
    print(name)

    q0 = Function(FunctionSpace(msh, "DG", 1))
    load_chk_point("./InitialEnsemble/64by64/particle_{}_0".format(_job), q0)

    particle.Particle(time_interval, _dt, msh, _job, _eof_dir, _var_).solver(dumpf, q0, name, _output_visual_flag=True, _chkpt_flag=True)


def run_spde_ensemble(_output_dir, _outer_range=20, _cores=25):
    """
    move initial ensemble forward using the spde

    :param _output_dir: where to store the output particle initial ensemble
    :param _outer_range: controls the num of particles
    :param _cores: number of cores
    """
    _res = 64
    _var = 0.5
    _eof_dir = "./eof/64by64/"
    msh = get_coarse_mesh(_res)
    _dt = get_coarse_dt(_res) / 2.

    time_interval = _dt     # << change this to run the solver for longer [0, time_interval]
    dumpf = int(round(eddy_turnover_time / _dt))    # output solution at every eddy turnover time ~ dt = 2.5 hardcoded

    for batch in [[x+_cores*y for x in range(_cores)] for y in range(_outer_range)]:
        procs = []
        for job in batch:
            params = [time_interval, dumpf, msh, _dt, job, _eof_dir, _var, _output_dir]
            myargs = (params, )
            proc = Process(target=wrapper, args=myargs)
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()


def run_ensemble(_cores,  _outer_range):
    print('begin {}'.format(64))
    eof_dir = './eof/64by64/'           ## eof file directory
    ensemble_output_dir = './output/'       ## output directory for data and visualisation files
    run_spde_ensemble(ensemble_output_dir, _outer_range, _cores)


if __name__ == "__main__":
    
    ## run n batches of m particles in parallel
    ## n = _cores
    ## m = _outer_range
    run_ensemble(_cores = 1, _outer_range = 1)  

