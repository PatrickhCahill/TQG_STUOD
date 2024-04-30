
#!/bin/bash

source "/home/wpan1/Software/firedrake_wei/bin/activate"

_nbatches=1
_nprocs=25

for ((i=0; i<_nbatches; ++i))
do
    echo "batch $i"
    mpiexec -n $_nprocs python /home/wpan1/Development/PythonProjects/ThermalQG/run_generate_initial_ensemble.py $i
done

deactivate




