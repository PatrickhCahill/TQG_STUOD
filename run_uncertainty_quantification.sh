#!/bin/bash

source "/home/wpan1/Software/firedrake_wei/bin/activate"

_max_step=300
_switch_off_bathymetry=0  # 1 for true, 0 for false
for ((step=0; step<_max_step; ++step))
do
    _nbatches=1
    _nprocs=25

    echo "step $step"

    # generate initial ensemble for current step
    for ((i=0; i<_nbatches; ++i))
    do
        echo "step $step, batch $i"
        # need to pass argument to the following in order to generate initial ensemble around a given truth
        mpiexec -n $_nprocs python /home/wpan1/Development/PythonProjects/ThermalQG/run_generate_initial_ensemble.py $i $step $_switch_off_bathymetry
    done

    # generate ensemble members
    # need to collect the data
    python /home/wpan1/Development/PythonProjects/ThermalQG/run_uncertainty_quantification_bathymetry_xi.py $step
       
done

#deactivate




