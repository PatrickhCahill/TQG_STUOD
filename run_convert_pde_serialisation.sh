#!/bin/bash

source "/home/wpan1/Software/firedrake/bin/activate"

# perturb
#mpiexec -n 16 python /home/wpan1/Development/PythonProjects/ThermalQG/convert_pde_serialisation.py 1

# no perturb
mpiexec -n 16 python /home/wpan1/Development/PythonProjects/ThermalQG/convert_pde_serialisation.py 0

deactivate

